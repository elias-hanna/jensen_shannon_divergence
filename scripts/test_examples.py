from jsd import KLdivergence
from jsd import JSdivergence

import numpy as np

def format_data(traj_data, action_data):
    ## Get number of samples
    num_samples_t = traj_data.shape[0]*traj_data.shape[1]
    num_samples_a = action_data.shape[0]*action_data.shape[1]
    assert num_samples_t == num_samples_a
    num_samples = num_samples_t
    ## Get A X S dimension
    s_dim = traj_data.shape[2]
    a_dim = action_data.shape[2] 
    as_dim = a_dim + s_dim
    ## Reshape input data
    reshaped_traj_data = np.reshape(traj_data, (num_samples, s_dim))
    reshaped_action_data = np.reshape(action_data, (num_samples, a_dim))
    ## Create empty container with proper dimensions
    formatted_data = np.empty((num_samples, as_dim))
    formatted_data[:,:a_dim] = reshaped_action_data
    formatted_data[:,a_dim:a_dim+s_dim] = reshaped_traj_data

    return formatted_data

if __name__ == '__main__':
    ## Load test data
    ra_data = np.load('example_data/ball_in_cup_random-actions_20_data.npz')
    rp_data = np.load('example_data/ball_in_cup_random-policies_20_data.npz')
    cnrw_0_data = np.load('example_data/ball_in_cup_colored-noise-beta-0_20_data.npz')
    cnrw_2_data = np.load('example_data/ball_in_cup_colored-noise-beta-2_20_data.npz')
    ## Get trajectory data and action data
    # ra_as = np.empty(())
    ## Reshape datasets
    f_ra_data = format_data(ra_data['train_trajs'], ra_data['train_actions'])
    f_rp_data = format_data(rp_data['train_trajs'], rp_data['train_actions'])
    f_cnrw_0_data = format_data(cnrw_0_data['train_trajs'], cnrw_0_data['train_actions'])
    f_cnrw_2_data = format_data(cnrw_2_data['train_trajs'], cnrw_2_data['train_actions'])

    ## Compute Kullback-Leibler and Jensen-Shannon Divergence between examples data sets
    # RA vs RP
    print(f"KLD(RA||RP)={KLdivergence(f_ra_data, f_rp_data)} | " \
          f"KLD(RP||RA)={KLdivergence(f_rp_data, f_ra_data)} | " \
          f"JSD(RP||RA)=JSD(RA||RP)={JSdivergence(f_ra_data, f_rp_data)}")

    # RA vs CNRW0
    print(f"KLD(RA||CNRW_0)={KLdivergence(f_ra_data, f_cnrw_0_data)} | " \
          f"KLD(CNRW_0||RA)={KLdivergence(f_cnrw_0_data, f_ra_data)} | " \
          f"JSD(RA||CNRW_0)=JSD(CNRW_0||RA)={JSdivergence(f_cnrw_0_data, f_ra_data)}")

    # RA vs CNRW2
    print(f"KLD(RA||CNRW_2)={KLdivergence(f_ra_data, f_cnrw_2_data)} | " \
          f"KLD(CNRW_2||RA)={KLdivergence(f_cnrw_2_data, f_ra_data)} | " \
          f"JSD(RA||CNRW_2)=JSD(CNRW_2||RA)={JSdivergence(f_cnrw_2_data, f_ra_data)}")

    # RP vs CNRW0
    print(f"KLD(RP||CNRW_0)={KLdivergence(f_rp_data, f_cnrw_0_data)} | " \
          f"KLD(CNRW_0||RP)={KLdivergence(f_cnrw_0_data, f_rp_data)} | " \
          f"JSD(RP||CNRW_0)=JSD(CNRW_0||RP)={JSdivergence(f_cnrw_0_data, f_rp_data)}")

    # RP vs CNRW2
    print(f"KLD(RP||CNRW_2)={KLdivergence(f_rp_data, f_cnrw_2_data)} | " \
          f"KLD(CNRW_2||RP)={KLdivergence(f_cnrw_2_data, f_rp_data)} | " \
          f"JSD(RP||CNRW_2)=JSD(CNRW_2||RP)={JSdivergence(f_cnrw_2_data, f_rp_data)}")
