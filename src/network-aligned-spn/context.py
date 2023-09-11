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