import numpy as np
import utils



# ========================== Mean =============================
def calculate_mean_shape(kpts):
    # ToDO
    pass



# ====================== Main Step ===========================
def procrustres_analysis_step(kpts, reference_mean):
    # Happy Coding
    pass



# =========================== Error ====================================

def compute_avg_error(kpts, mean_shape):
    # ToDo
    pass




# ============================ Procrustres ===============================

def procrustres_analysis(kpts, max_iter=int(1e3), min_error=1e-5):

    aligned_kpts = kpts.copy()

    for iter in range(max_iter):

        reference_mean = calculate_mean_shape(aligned_kpts)

        # align shapes to mean shape
        aligned_kpts = procrustres_analysis_step(aligned_kpts, reference_mean)

        ##################### Your Part Here #####################

        ##########################################################


    # visualize

    # visualize mean shape

    return aligned_kpts
