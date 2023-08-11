from spn.structure.Base import Context
from spn.structure.StatisticalTypes import MetaType

ctxts = {
    'income': {
        i: MetaType.REAL for i in range(14)
    }
}

ctxts['income'][14] = MetaType.BINARY