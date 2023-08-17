from spn.structure.Base import Context
from spn.structure.StatisticalTypes import MetaType

ctxts = {
    'income': {
        i: MetaType.REAL for i in range(14)
    },
    'mnist': {
        i: MetaType.REAL for i in range(28*28)
    }
}

ctxts['income'][14] = MetaType.BINARY
ctxts['income'][15] = MetaType.BINARY
ctxts['mnist'][(28*28)] = MetaType.DISCRETE