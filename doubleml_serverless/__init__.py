from pkg_resources import get_distribution

from .double_ml_plr_aws_lambda import DoubleMLPLRServerless
from .double_ml_pliv_aws_lambda import DoubleMLPLIVServerless
# from .double_ml_irm_aws_lambda import DoubleMLIRMServerless
# from .double_ml_iivm_aws_lambda import DoubleMLIIVMServerless
from .double_ml_data_aws import DoubleMLDataS3, DoubleMLDataJson

__version__ = get_distribution('doubleml').version
