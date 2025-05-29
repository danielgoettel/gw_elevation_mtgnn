import numpy as np
import pastastore as pst

def msuffix_to_description(msuffix):
    riv_opts = {
        "0": "",
        "1": "riv"
    }
    well_opts = {
        "0": "",
        "1": "wells1",
        "2": "wells2",
        "3": "wells3"
    }

    parts = ["rech", riv_opts[msuffix[1]], well_opts[msuffix[2]]]
    parts = [i for i in parts if len(i) > 0]

    return "_".join(parts)


def variance_gain(ml, wm_name, istress=None):
    """
    https://pastas.readthedocs.io/en/latest/concepts/hantush_response.ipynb.html

    """
    wm = ml.stressmodels[wm_name]

    if ml.fit is None:
        raise AttributeError("Model not optimized! Run solve() first!")
    if wm.rfunc._name != "HantushWellModel":
        raise ValueError("Response function must be HantushWellModel!")

    # get parameters and (co)variances
    A = ml.parameters.loc[wm_name + "_A", "optimal"]
    b = ml.parameters.loc[wm_name + "_b", "optimal"]
    var_A = ml.fit.pcov.loc[wm_name + "_A", wm_name + "_A"]
    var_b = ml.fit.pcov.loc[wm_name + "_b", wm_name + "_b"]
    cov_Ab = ml.fit.pcov.loc[wm_name + "_A", wm_name + "_b"]

    if istress is None:
        r = np.asarray(wm.distances)
    elif isinstance(istress, int) or isinstance(istress, list):
        r = wm.distances[istress]
    else:
        raise ValueError("Parameter 'istress' must be None, list or int!")

    return wm.rfunc.variance_gain(A, b, var_A, var_b, cov_Ab, r=r)

def get_pastastore(name, connector_type):
    if connector_type == "arctic":
        connstr = 'mongodb://localhost:27017/'
        conn = pst.ArcticConnector(name, connstr)
    elif connector_type == "pas":
        path = "C:\\Users\\danielg\\Pycharm Projects\\Taccari_et_Al_Complete\\GroundwaterFlowGNN-main\\pastas_db"
        conn = pst.PasConnector(name, path)
    return pst.PastaStore(name, conn)