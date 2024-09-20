from spn.structure.Base import Context
from spn.structure.StatisticalTypes import MetaType

ctxts = {
    'income': {
        i: MetaType.REAL for i in range(14)
    },
    'mnist': {
        i: MetaType.REAL for i in range(28*28)
    },
    'avazu': {
        i: MetaType.REAL for i in range(1, 26)
    },
    'credit': {
        i: MetaType.DISCRETE for i in range(5, 9)
    },
    'breast-cancer': {
        i: MetaType.REAL for i in range(30)
    },
    'baf': {
        i: MetaType.REAL for i in [0, 1, 5, 6, 9, 10, 11, 22, 25]
    }
}

ctxts['income'][14] = MetaType.BINARY
ctxts['income'][15] = MetaType.BINARY

ctxts['mnist'][(28*28)] = MetaType.DISCRETE

ctxts['avazu'][0] = MetaType.BINARY

ctxts['credit'][9] = MetaType.REAL
ctxts['credit'][10] = MetaType.BINARY
ctxts['credit'][0] = MetaType.REAL
ctxts['credit'][1] = MetaType.DISCRETE
ctxts['credit'][2] = MetaType.DISCRETE
ctxts['credit'][3] = MetaType.REAL
ctxts['credit'][4] = MetaType.REAL

ctxts['breast-cancer'][30] = MetaType.BINARY

ctxts['baf'][16] = MetaType.BINARY
ctxts['baf'][18] = MetaType.BINARY
ctxts['baf'][19] = MetaType.BINARY
ctxts['baf'][21] = MetaType.BINARY
ctxts['baf'][23] = MetaType.BINARY
ctxts['baf'][24] = MetaType.BINARY
ctxts['baf'][27] = MetaType.BINARY
ctxts['baf'][30] = MetaType.BINARY
for i in range(30):
    if i not in list(ctxts['baf'].keys()):
        ctxts['baf'][i] = MetaType.DISCRETE


feature_types = {
    'income': {
        i: 'cont' for i in range(14)
    },
    'mnist': {
        i: 'cont' for i in range(28*28)
    },
    'avazu': {
        i: 'cont' for i in range(1, 26)
    },
    'credit': {
        i: 'ord' for i in range(5, 9)
    },
    'breast-cancer': {
        i: 'cont' for i in range(30)
    }
}

feature_types['income'][14] = 'cat'
feature_types['income'][15] = 'cat'
feature_types['mnist'][(28*28)] = 'ord'
feature_types['avazu'][0] = 'cat'
feature_types['credit'][9] = 'cont'
feature_types['credit'][10] = 'cat'
feature_types['credit'][0] = 'cont'
feature_types['credit'][1] = 'ord'
feature_types['credit'][2] = 'ord'
feature_types['credit'][3] = 'cont'
feature_types['credit'][4] = 'cont'
feature_types['breast-cancer'][30] = 'cat'