#!/usr/bin/python3

# Performs ADF and KPSS test on a set of reflections to determine whether they are stationary or 
# ADF_test.py can be run using expt and refl file OR on json file previously made using this script
# Use scaled.expt and scaled.refl files only
# Will generate a ADF_KPSS_test.log file and a new refl file containing outputs from ADF and KPSS testing

# Example: python ADF_test.py --expt_fp /data/scaled.expt --refl_fp /data/scaled.refl --json_fp /data/adf_test_data.json
# my own use: python3 ADF_KPSS_test.py --expt_fp /dls/mx-scratch/rachelt/project/disulphide/scaled.expt --refl_fp /dls/mx-scratch/rachelt/project/disulphide/scaled.refl --i_s=10 --autolag_type AIC --alpha=0.05
# null dataset: python3 ADF_KPSS_test.py --expt_fp /dls/mx-scratch/rachelt/project/disulphide/null_dataset/data/scaled.expt --refl_fp /dls/mx-scratch/rachelt/project/disulphide/null_dataset/data/scaled.refl --i_s=10 --autolag_type AIC --alpha=0.05

# basic libraries
import os
import numpy as np
import json
import pandas as pd
import argparse
import logging 

from dxtbx import flumpy
from dxtbx.model.experiment_list import ExperimentList
from dxtbx.model import experiment_list
from dials.array_family import flex

# some useful tools from scaling which will help our analysis here
from dials.algorithms.scaling.Ih_table import map_indices_to_asu

# statistical library
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss

def evalulate_adf_stat(stat, crit):
    # evalulate the adf test statistic against the critical value
    if stat < crit["1%"]:
        level = "1%"
        num_level = 1
    elif stat < crit["5%"]:
        level = "5%"
        num_level = 5
    elif stat < crit["10%"]:
        level = "10%"
        num_level = 10
    else:
        # the series likely has a unit root (non-stationary)
        level = "non-stationary"
        num_level = 100
    return level, num_level

def evaluate_kpss_stat(stat, crit):
    # evalulate the kpss test statistic against the critical value
    if stat > crit["1%"]:
        level = "non-stationary at 1%"
        num_level = 1
    elif stat > crit["5%"]:
        level = "non-stationary at 5%"
        num_level = 5
    elif stat > crit["10%"]:
        level = "non-stationary at 10%"
        num_level = 10
    else:
        # the series is likely stationary
        level = "stationary"
        num_level = 100
    return level, num_level

def get_pval_result(pval, alpha=0.05, test="adf"):
    # evaluating a the p values from adf or kpss
    # note that the null hypothesis is non-stationary for adf
    # whilst the null hypothesis is stationary for kpss
    if test.lower() == "adf":
        if pval < alpha:
            result = "stationary"
            num_res = 0
        else:
            result = "non-stationary"
            num_res = 1
        return result, num_res
    elif test.lower() == "kpss":
        if pval < alpha:
            result =  "non-stationary" 
            num_res = 1
        else:
            result = "stationary"
            num_res = 0
        return result, num_res
    else:
        raise ValueError("test must be 'adf' or 'kpss'")

# Sample: grouped on 'hkl', time-like column 'peak_loc', value column 'intensity'
def adf_kpss_per_group(df, group_col="hkl", time_col="peak_loc", value_col="intensity", autolag_setting="AIC", regression='ct', alpha=0.05):
    results = []
    count = 0 

    # generating a time series dataset of all obsesrvation of the same hkl
    # sorted by the peak location
    grouped = df.sort_values(time_col).groupby(group_col)

    for name, group in grouped:
        # intensity values for each hkl 
        y = group[value_col].values
        try:
            # Only test if enough values (ADF uses lags)
            if len(y) > 8:
                # adf test function from statsmodels.tsa.stattools
                adf_result = adfuller(y, autolag=autolag_setting, regression='ct')
                # save result values
                adf_stat = adf_result[0]
                adf_pval = adf_result[1]
                adf_crit = adf_result[4]

                # if adf_stat < critical_value then reject null hypothesis (data is stationary)
                # alternatively if If adf_stat > critical_value, then fail to reject null (data may have a unit root, non-stationary).
                adf_level, adf_num_level = evalulate_adf_stat(adf_stat, adf_crit)
                adf_pval_result, adf_pval_result_num  = get_pval_result(adf_pval, alpha, test="adf")

                # KPSS Test from statsmodels.tsa.stattools library
                kpss_result = kpss(y, regression="ct", nlags="auto")
                # save result values
                kpss_stat = kpss_result[0]
                kpss_pval = kpss_result[1]
                kpss_crit = kpss_result[3]

                # evalulate the results
                kpss_level, kpss_num_level  = evaluate_kpss_stat(kpss_stat, kpss_crit)
                kpss_pval_result, kpss_pval_result_num = get_pval_result(kpss_pval, 0.05, test="kpss")

                # evaluating both pvalues to determine the degree of confidence 
                # and consistency between results
                if adf_pval < 0.05 and kpss_pval > 0.05:
                    interpretation = "stationary"
                    num = 0
                elif adf_pval > 0.05 and kpss_pval < 0.05:
                    interpretation = "non-stationary"
                    num = 1
                elif adf_pval > 0.05 and kpss_pval > 0.05:
                    interpretation = "inconclusive"
                    num = 2
                elif adf_pval < 0.05 and kpss_pval < 0.05:
                    interpretation = "trend-stationary or mixed"
                    num = 3
                else:
                    interpretation = "unknown"
                    num = 4 

                results.append(
                    {
                        group_col: name,
                        "adf_statistic": adf_stat,
                        "adf_p_value": adf_pval,
                        # p-value is less than 5%, reject null hypothesis (stationary)
                        "adf_pval_result": adf_pval_result,
                        "adf_pval_result_num": adf_pval_result_num,
                        "adf_stat_result": adf_level,
                        "adf_stat_result_num" :adf_num_level,
                        # Use this if you want to extract the 
                        # critical values and thresholds for future use
                        #"crit_vals": adf_result[4],
                        "kpss_statistic": kpss_stat,
                        "kpss_p_value": kpss_pval,
                        "kpss_pval_result": kpss_pval_result,
                        "kpss_pval_result_num": kpss_pval_result_num,
                        "kpss_stat_result": kpss_level,
                        "kpss_stat_result_num": kpss_num_level,
                        "adf_and_kpss_pvals_result": interpretation,
                        "adf_and_kpss_pvals_result_num": num,
                        "note": "",
                    }
                )
                count = count + 1 
        except Exception as e:
            results.append(
                {
                    group_col: name,
                    "adf_statistic": "none",
                    "adf_p_value": "none",
                    "adf_pval_result": "none",
                    "adf_pval_result_num": "none",
                    "adf_stat_result": "none",
                    "adf_stat_result_num" :"none",
                    "kpss_statistic": "none",
                    "kpss_p_value": "none",
                    "kpss_pval_result": "none",
                    "kpss_pval_result_num": "none",
                    "kpss_stat_result": "none",
                    "kpss_stat_result_num": "none",
                    "adf_and_kpss_pvals_result": "none",
                    "adf_and_kpss_pvals_result_num": "none",
                    "note": f"Error: {e}",
                }
            )

    return pd.DataFrame(results), count 


def setup_logger(log_file="ADF_KPSS_test.log", logger_name= "ADFLogger"):
    # Set up the logger called "ADFLogger" writing into "ADF_KPSS_test.log" file
    # Note the log is never overwritten, just appended to
    # if this pythong scipt is run again
    logger = logging.getLogger("ADFLogger") 
    # Set minimum level of logging 
    logger.setLevel(logging.INFO)

    # StreamHandler manages terminal messages 
    console_handler = logging.StreamHandler()
    # File handler manges messages to ADF_KPSS_test.log
    file_handler = logging.FileHandler("ADF_KPSS_test.log")

    # Create a format for each message 
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger 

#testing this fucntion to see if differencing can help with anything 
def adf_withdiff_per_group(df, group_col="hkl", time_col="peak_loc", value_col="intensity", autolag_setting="AIC", regression='ct'):
    results = []

    # sort by peak_loc and then group by hkl
    grouped = df.sort_values(time_col).groupby(group_col)

    for name, group in grouped:
        # intensity values for each hkl 
        y = group[value_col].values
        y_diff1 = pd.Series(np.diff(y))
        y_diff2 = y_diff1.diff().dropna()
        
        for i, series in enumerate([y, y_diff1, y_diff2], start=0):
            result = adfuller(series, regression='ct')
            print(f"Order of differencing: {i}")
            print(f"  ADF stat = {result[0]:.3f},  p‐value = {result[1]:.3f}")
            print(f"  1% cv = {result[4]['1%']:.3f}, 5% cv = {result[4]['5%']:.3f}")
            print()
            break

def run(args=None):
    logger = setup_logger()
    logger.info("Running ADF_KPSS_test.py")
    
    # Reads in the arguments provided in the terminal
    # Specifying expt, refl and json file paths ideally
    parser = argparse.ArgumentParser(
        description="Extracting data from either json file or .expt and .refl files."
    )
    parser.add_argument("--expt_fp", default=None, help="Path to the expt file")
    parser.add_argument("--refl_fp", default=None, help="Path to the refl file")
    parser.add_argument("--autolag_type", default="AIC", help="AIC, BIC or t-stat for the ADF testing")
    parser.add_argument("--i_s", default=10, type=float, help="i_s cutoff value")
    parser.add_argument("--alpha", default=0.05, type=float, help="alpha value for the statistical tests")

    logger.info("-----------------------------------------------------")
    parsed_args = parser.parse_args(args)
    logger.info(f"Parsed arguments are:")
    logger.info(f".expt filepath: {parsed_args.expt_fp}")
    logger.info(f".refl filepath: {parsed_args.refl_fp}")
    logger.info(f"Autolag type: {parsed_args.autolag_type}")
    logger.info(f"i_s cutoff set to: {parsed_args.i_s}")
    logger.info(f"alpha set to: {parsed_args.alpha}")
    i_s_threshold = parsed_args.i_s
    alpha = parsed_args.alpha
  
    # Checks that expt_fp and refl_fp have been provided 
    if (
        not parsed_args.expt_fp
        or not parsed_args.refl_fp
    ):
        logger.warning("Please provide .expt and .refl file paths")
        logger.warning(
            "Example: python ADF_test.py --expt_fp /data/test.expt --refl_fp /data/test.refl --i_s=10 --autolag_type AIC"
        )
        logger.error("Please provide filepaths")
    else:
        expt_fp = parsed_args.expt_fp
        refl_fp = parsed_args.refl_fp
    
    # Determines auto lag type from args, if none given then AIC is used
    if not parsed_args.autolag_type:
        logger.info("Autolag type not spcified, autolag set to default AIC")
    else: 
        autolag_type = parsed_args.autolag_type
        permitted_autolag_types = ["AIC", "BIC", "t-stat"]
        if autolag_type in permitted_autolag_types: 
            pass
        else: 
            logger.warning("Chosen autolag type is not permitted")
            logger.warning("Autolag type changed to default AIC")
            logger.warning("Next time. please select from list: AIC, BIC, t-stat")
            logger.warning("Example input: --autolag_type AIC")
            autolag_type = "AIC"

    # Adds a new column to a refl table and handles the datatype
    def new_refl_column(all_refls, flumpy_index, column_name, flex_type = flex.double, data_type = "d"):
        # The column inititalised with values of 200.0 in each row
        # 200.0 is the "nan" value
        all_refls[column_name] = flex_type(all_refls.size(), 200.0)
        new_data = flumpy.from_numpy(np.array(df[column_name], dtype= data_type))
        all_refls[column_name].set_selected(flumpy_index, new_data)
        return all_refls

    logger.info("-----------------------------------------------------")
    logger.info(f"Opening expt and refl files")
    hkl_list, intensity_list, peak_loc_list, i_s_list, index_list = [], [], [], [], []
    
    # Opening expt and refl files, obtained from standard dials processing
    expts = ExperimentList.from_file(expt_fp)
    all_refls = flex.reflection_table.from_file(refl_fp)
    # Generating an index column to enable writing data back into the refl table
    all_refls["index_refl"] = flumpy.from_numpy(np.array(list(range(0,len(all_refls))), dtype='uint64'))

    # Here we are selecting the _good_ reflections - these are the ones
    # which are flagged as (i) having been scaled and (ii) _not_ flagged as
    # having been bad for scaling - the flags are packed into a bit array
    # - then we _invert_ the selection of bad for scaling reflections with ~
    refls = all_refls.select(all_refls.get_flags(all_refls.flags.scaled))
    refls = refls.select(~refls.get_flags(refls.flags.bad_for_scaling, all=False))

    # Map the reflections to the asymmetric unit, make a new column with
    # the unique Miller index as asu_miller_index
    space_group = expts[0].crystal.get_space_group()
    refls["asu_miller_index"] = map_indices_to_asu(refls["miller_index"], space_group)
    unique_hkl = set(refls["asu_miller_index"])
    for hkl in unique_hkl:
        # Selects reflections that matches the unique hkl
        refl0 = refls.select(refls["asu_miller_index"] == hkl)
        intensities = refl0["intensity.scale.value"] / refl0["inverse_scale_factor"]
        # inverse_scale_factor = flumpy.to_numpy(refl0["inverse_scale_factor"])
        # Divide intensity.scale.value by inverse_scale_factor to obtain the scaled intensities
        sigma = flex.sqrt(refl0["intensity.scale.variance"]) / refl0["inverse_scale_factor"]
        # Compute the mean I/σ(I)
        i_s = flex.mean(intensities / sigma)
        intensities = flumpy.to_numpy(intensities)
        # This gives the list of calculated peak position in radians for relfections == hkl
        peak_locations = list(refl0["xyzcal.mm"].parts()[2])
        index_val = refl0["index_refl"]
        for i, intensity in enumerate(intensities):
            peak_loc = np.degrees(peak_locations[i])
            intensity_list.append(intensity)
            hkl_list.append(hkl)
            peak_loc_list.append(peak_loc)
            i_s_list.append(i_s)
            index_list.append(index_val[i])
    
    # Writing the lists to a dictionary to convert into a json file
    temp_dict = {
        "hkl": hkl_list,
        "intensity": intensity_list,
        "i_s": i_s_list,
        "peak_loc": peak_loc_list,
        "index_val": index_list
    }

    df = pd.DataFrame(data=temp_dict)

    # A high I/sigma value indicates that the intensity measurement is well-determined and reliable,
    # meaning the signal is strong compared to the noise
    df = df.sort_values(by=["hkl", "peak_loc"])
    df = df.reset_index(drop=True)
    count_all_obs = len(df)
    # remove datasets with high values or i/sigma
    df = df[df["i_s"] > i_s_threshold]
    if len(df) == 0:
        logger.error(f"problem accessing the data, please check expt and refl files, or the json file")
    logger.info(
        f"{count_all_obs-len(df):,} peaks removed from analysis due to low i/sigma (less than 10)"
    )
    logger.info(f"There are {len(df):,} remaining observations/ rows (total number of recorded peaks)")

    # This command calls the adf_kpss_per_group function that runs the ADF and KPSS tests
    # on each hkl group in the df as its own time series dataset
    # When regression='ct', it tests if the dataset has a unit root with drift and/or trend
    # results_df contains the results where eack hkl occupies one row
    results_df, num_hkls = adf_kpss_per_group(
        df, group_col="hkl", time_col="peak_loc", value_col="intensity", autolag_setting=autolag_type, regression='ct'
    )

    # Prints out some facts about the results from ADF and KPSS tests
    logger.info("-----------------------------------------------------")
    logger.info(f"{num_hkls:,} of total hkls analysed")
    logger.info("-----------------------------------------------------")
    
    logger.info("TEST STATISTIC AND CRITICAL VALUES")

    count = (results_df["adf_stat_result"] == "non-stationary").sum()
    logger.info(f"ADF: {count:,} non-stationary hkls ({count/num_hkls:0.0%})")

    count = (results_df["kpss_stat_result"] != "stationary").sum()
    logger.info(f"KPSS: {count:,} non-stationary hkls at the 10% level or greater ({count/num_hkls:0.0%})")

    count = ((results_df['adf_stat_result'] == "non-stationary") & (results_df['kpss_stat_result'] != "stationary")).sum()
    logger.info(f"Combining ADF and KPSS: {count:,} non-stationary hkls ({count/num_hkls:0.0%})")

    logger.info("-----------------------------------------------------")
    logger.info("P_VALUE RELATED RESULTS")
    logger.info(f"Alpha level used: {alpha} (this affects the p value related results)")
    count = (results_df["adf_pval_result"] == "non-stationary").sum()
    logger.info(f"ADF: {count:,} non-stationary hkls ({count/num_hkls:0.0%})")

    count = (results_df["kpss_pval_result"] == "non-stationary").sum()
    logger.info(f"KPSS: {count:,} non-stationary hkls ({count/num_hkls:0.0%})")

    count = (results_df["adf_and_kpss_pvals_result"] == "non-stationary").sum()
    logger.info(f"Evalulating ADF and KPSS: {count:,} non-stationary hkls ({count/num_hkls:0.0%})")

    count = (results_df["adf_and_kpss_pvals_result"] == "trend-stationary or mixed").sum()
    logger.info(f"Evalulating ADF and KPSS: {count:,} trend-stationary or mixed hkls ({count/num_hkls:0.0%})")

    count = (results_df["adf_and_kpss_pvals_result"] == "inconclusive").sum()
    logger.info(f"Evalulating ADF and KPSS: {count:,} inconclusive hkls ({count/num_hkls:0.0%})")

    logger.info("-----------------------------------------------------")
    logger.info("SUBSET OF DATA: flagged as non-stationary using test statistics and critical values")
    #logger.info(f"ADF stat shows non-stationary AND KPSS stats also shows non-stationary (regardless of level)")
    logger.info("A small p-value (typically less than 0.05) indicates strong evidence against the null hypothesis")
    subset = results_df[
    (results_df['adf_stat_result'] == "non-stationary") &
    (results_df['kpss_stat_result'] != "stationary")    ]

    adf_range = subset["adf_p_value"].min(), subset["adf_p_value"].max()
    if (adf_range[0] > alpha) and (adf_range[1] > alpha):
        logger.info(f"Range of ADF p-values: {adf_range[0]:.3} - {adf_range[1]:.3}, all p values indicate non-stationarity")
    else: 
        logger.info(f"Range of ADF p-values: {adf_range[0]:.3} - {adf_range[1]:.3}, max value greater than alpha: ({alpha}) which may suggest stationarity at the given alpha level")

    kpss_range = subset["kpss_p_value"].min(), subset["kpss_p_value"].max()

    if (kpss_range[0] < alpha) and (kpss_range[1] < alpha):
        logger.info(f"Range of KPSS p-values: {kpss_range[0]:.3} - {kpss_range[1]:.3}, all p values indicate non-stationarity")
    else: 
        logger.info(f"Range of KPSS p-values: {kpss_range[0]:.3} - {kpss_range[1]:.3}, min value greater than alpha ({alpha}) which may suggest stationarity at the given alpha level")
    logger.info("Note: With kpss sometimes the actual p-value is greater than the p-value returned as the test statistic is outside of the range of p-values available in the look-up table.")
    
    # it would be interesting to plot all p values for each test?
    # it would also be interesting to filter it by only hkls which are lagged as non-stationary using test statistics and critical values?

    logger.info("-----------------------------------------------------")

    # Merge the results onto the main df so that results can be written into a new refl file
    df = df.merge(results_df, on='hkl', how='left')

    # Making a flumpy list of index values as they appear in the df
    flumpy_index = flumpy.from_numpy(np.array(df["index_val"], dtype="uint64"))
    
    # Only numerical values can be moved into refl file
    numerical_column_name_list = ['adf_statistic', 'adf_p_value','adf_pval_result_num', 'adf_stat_result_num', 
                                  'kpss_statistic', 'kpss_p_value','kpss_pval_result_num', 'kpss_stat_result_num', 
                                  'adf_and_kpss_pvals_result_num']
    
    for name in numerical_column_name_list: 
        all_refls = new_refl_column(all_refls, flumpy_index, column_name=name, flex_type = flex.double, data_type = "d")

    all_refls.as_file(os.path.splitext(refl_fp)[0] +"_ADF_KPSS.refl")

    logger.info(f"New columns added to refl file: {numerical_column_name_list}")
    logger.info(f"New refl file: {os.path.splitext(refl_fp)[0]}_ADF_KPSS.refl")
    logger.info("a value of 200 indicates a nan value")
    logger.info("in most columns 0 indicates stationary whilst 1 indicates non-stationary")
    logger.info("adf_num_level 100 indicates non-stationary")
    logger.info("kpss_num_level 100 indicates stationary")
    logger.info("adf_and_kpss_pvals_result_num 2 indicates inconclusive, 3 indicates trend-stationary or mixed whilst 4 indicates unknown")
    logger.info("-----------------------------------------------------")
    logger.info("One of the datasets in your analysis:")
    logger.info(results_df.iloc[0])
    logger.info("-----------------------------------------------------")
    logger.info("END")

if __name__ == "__main__":
    run()
