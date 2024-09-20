import numpy as np
from spn.algorithms.LearningWrappers import learn_mspn
from spn.structure.Base import Context
from spn.structure.StatisticalTypes import MetaType
from spn.algorithms.Inference import log_likelihood

class DecisionNode:
    def __init__(self, depth, split_feature, split_feature_type, split_val) -> None:
        self.depth = depth
        self.split_feature = split_feature
        self.split_feature_type = split_feature_type
        self.split_val = split_val
        self.left: LeafNode | DecisionNode
        self.right: LeafNode | DecisionNode

    def condition(self, x):
        if self.split_feature_type == 'cat':
            return x[self.split_feature] == self.split_val
        else:
            return x[self.split_feature] <= self.split_val

class LeafNode:
    def __init__(self, data, depth, feature_types) -> None:
        self.data = data
        self.depth = depth
        self.feature_types = feature_types

    def density(self, x, N):
        raise NotImplementedError()

class NonParamLeafNode(LeafNode):

    def __init__(self, data, depth, feature_types) -> None:
        super().__init__(data, depth, feature_types)

    def density(self, x, N):
        """
            Compute (piecewise constant) denisty of a certain region of the feature space represented by this node.
            Follows approach of https://mlpack.org/papers/det.pdf
            NOTE: Currently not exact, requires decision path for continuous features
            TODO: Think about making exact if performance not good enough
        """

        mins = np.array([self.data[:, i].min() for i in range(len(self.feature_types)) if self.feature_types[i] == 'cont'])
        maxs = np.array([self.data[:, i].max() for i in range(len(self.feature_types)) if self.feature_types[i] == 'cont']) + 1e-6 # ensure non-zero difference
        num_cats = np.array([len(np.unique(self.data[:, i])) for i in range(len(self.feature_types)) if self.feature_types[i] == 'cat'])
        ord_mins = np.array([self.data[:, i].min() for i in range(len(self.feature_types)) if self.feature_types[i] == 'ord'])
        ord_maxs = np.array([self.data[:, i].max() for i in range(len(self.feature_types)) if self.feature_types[i] == 'ord'])

        vol_rd = np.prod(maxs - mins).item()
        prod_cats = np.prod(num_cats).item()
        ord_vol = np.prod(ord_maxs - ord_mins).item()

        return (len(self.data) * N) / (vol_rd * prod_cats * ord_vol)
    
class HistLeafNode(LeafNode):

    def __init__(self, data, depth, feature_types, num_bins=5) -> None:
        super().__init__(data, depth, feature_types)
        self.np_hist, self.np_edges = None, None
        self.c_edges = []
        self.data = data
        self.num_bins = num_bins
        self._create_histogram(data)
        print([np.sum(data[:, -1] == 0), np.sum(data[:,-1] == 1)])

    def density(self, x, N):
        if self.np_edges is not None:
            return self._np_density(x, N)
        elif self.np_edges is None and self.c_edges is not None:
            return self._c_density(x, N)
        else:
            raise RuntimeError('Could not fit leaf')
        
    def _np_density(self, x, N):
        density_idx = [0]*len(x)
        indices_assigned = [False]*len(x)
        for i, xi in enumerate(x):
            # for each dimension, find bin where xi fits in
            dim_edges = self.edges[i]
            for j in range(1, len(dim_edges)):
                if xi >= dim_edges[j-1] and xi < dim_edges[j]:
                    density_idx[i] = j - 1
                    indices_assigned[i] = True
        if all(indices_assigned):
            density = (len(self.data) / N) * self.hist[*density_idx]
            density = density if density != 0 else 1e-9
        else:
            return 1e-9
        

    def _c_density(self, x, N):
        fdata = np.copy(self.data)
        widths = []
        for i, xi in enumerate(x):
            dim_edges = self.c_edges[i]
            if self.feature_types[i] == 'cont':
                for j in range(1, len(dim_edges)):
                    if dim_edges[j-1] <= xi and xi < dim_edges[j]:
                        l, r = dim_edges[j - 1], dim_edges[j]
                        dim_data = fdata[:, i]
                        idx = np.argwhere((dim_data >= l) & (dim_data < r)).flatten()
                        #print(f"FILTER. DIM: {i}: L;R=({l}, {r})")
                        #print(fdata.shape)
                        fdata = fdata[idx]
                        #print(fdata.shape)
                        widths.append(l - r)
            else:
                if xi in dim_edges:
                    dim_data = fdata[:, i]
                    idx = np.argwhere(dim_data == xi).flatten()
                    if self.feature_types[i] == 'ord':
                        width = dim_edges.max() - dim_edges.min()
                    else:
                        width = len(np.unique(dim_edges))
                    fdata = fdata[idx]
                    widths.append(width)
        if len(widths) == fdata.shape[1]:
            #print(fdata.shape)
            density = (len(self.data) / N) * (len(fdata) / np.prod(widths).item())
            density = density if density != 0 else 1e-9
            return density
        else:
            return 1e-9 # return non-zero density to avoid 0-densities

    #def _c_density(self, x, N):
    #    fdata = np.copy(self.data)
    #    widths = []
    #    cond_indices = []
    #    for i, xi in enumerate(x):
    #        dim_edges = self.c_edges[i]
    #        if self.feature_types[i] == 'cont':
    #            for j in range(1, len(dim_edges)):
    #                if dim_edges[j-1] <= xi and xi < dim_edges[j]:
    #                    l, r = dim_edges[j - 1], dim_edges[j]
    #                    dim_data = fdata[:, i]
    #                    idx = np.argwhere((dim_data >= l) & (dim_data < r)).flatten()
    #                    cond_indices.append(idx)
    #                    widths.append(l - r)
    #        else:
    #            if xi in dim_edges:
    #                dim_data = fdata[:, i]
    #                idx = np.argwhere(dim_data == xi).flatten()
    #                if self.feature_types[i] == 'ord':
    #                    width = dim_edges.max() - dim_edges.min()
    #                else:
    #                    width = len(np.unique(dim_edges))
    #                cond_indices.append(idx)
    #                widths.append(width)
    #    if len(widths) == fdata.shape[1]:
    #        cond_indices = [set(idx.tolist()) for idx in cond_indices]
    #        print("SETS")
    #        print("".join([f"{s} \n" for s in cond_indices]))
    #        intersection = cond_indices[0].intersection(*cond_indices)
    #        box_samples = len(intersection)
    #        density = (len(self.data) / N) * (box_samples / np.prod(widths).item())
    #        density = density if density != 0 else 1e-9
    #        return density
    #    else:
    #        return 1e-9 # return non-zero density to avoid 0-densities
        
        
    def _create_histogram(self, data: np.ndarray):
        num_bins = []
        for i, ft in enumerate(self.feature_types):
            if ft == 'cont':
                nb = self.num_bins
                num_bins.append(nb) # simply set 10 bins for continuous features
            elif ft == 'ord' or ft == 'cat':
                num_bins.append(len(np.unique(data[:, i])))
        try:
            # try to do it with numpy
            raise ValueError()
            self.hist, self.edges = np.histogramdd(data, num_bins, density=True)
        except:
            # fails most likely due to memory issues, use alternative implementation
            for j in range(data.shape[1]):
                if self.feature_types[j] == 'cont':
                    dim_rng = data[:, j].max() - data[:, j].min()
                    bin_width = dim_rng / num_bins[j]
                    limits = [data[:, j].min() + i * bin_width for i in range(num_bins[j])]
                    self.c_edges.append(limits)
                else:
                    limits = np.unique(data[:, j])
                    self.c_edges.append(limits)
            
    
class PCLeafNode(LeafNode):

    def __init__(self, data, depth, feature_types) -> None:
        super().__init__(data, depth, feature_types)
        self.meta_types = []
        for ft in self.feature_types:
            if ft == 'cont':
                self.meta_types.append(MetaType.REAL)
            elif ft == 'ord':
                self.meta_types.append(MetaType.DISCRETE)
            elif ft == 'cat':
                self.meta_types.append(MetaType.BINARY)
        self.ctxt = Context(meta_types=self.meta_types)
        self.ctxt.add_domains(data)
        self.pc = learn_mspn(data, self.ctxt, min_instances_slice=10)

    def density(self, x, N):
        return log_likelihood(self.pc, x).flatten()