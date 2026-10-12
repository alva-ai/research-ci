from evals.toy import closes_at_zero

from research_ci.api import Eval

EVALS = [Eval(closes_at_zero, h0="every close is above zero")]
