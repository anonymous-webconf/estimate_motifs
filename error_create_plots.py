from matplotlib import pyplot as plt
import pandas as pd
import re
import numpy as np

csv_file = "dataset_error_details.csv"

error_detail_csv = pd.read_csv(csv_file, index_col=0)

alpha = 0.8
skip = 0

end = -1

m_style = "."

g_name_map = {"cyclic-C3": "cyclic-C3", "cyclic-C4" : "cyclic-C4", "wedge": "wedge", 
              "leaf-C3": "G24", "clique-K4":"clique-K4", "butterfly":"butterfly"}
fw_name_map = {"RWIS_G1": "RW-IS on "+r"$G$",
               "RWIS_G2": "RW-IS on "+r"$\mathcal{L}(\widehat{G})$",
               "RWRJ": "RW-RJ, "+r"$\alpha: {}$".format(alpha),
               "RWBB" : "RW-BB, "+r"$\alpha: {}$".format(alpha)}

def create_error_plot(df : pd.DataFrame, save_loc: str):
    # the first row of the dataframe is {dataset_name}_{gaphlet}
    # for each column there is one plot
    for col in list(df.columns):
        tem = col.split("_")
        assert len(tem)==2
        dataset_name = tem[0]
        graphlet_name = tem[1]

        fig = plt.figure(figsize=(4.5,3.5))

        # if graphlet_name == "cyclic-C4":
        #     G1_det = eval(df.loc["RWBB", col])
        #     plt.plot(G1_det["r"][skip:end], G1_det["error"][skip:end], label = "walk on G1", marker = m_style)
        #     plt.annotate(str(round(G1_det["error"][-1],3)), (G1_det["r"][-1], G1_det["error"][-1]))

        #     G2_det = eval(df.loc["RWRJ", col])
        #     plt.plot(G2_det["r"][skip:end], G2_det["error"][skip:end], label = "walk on G2", marker = m_style)
        #     plt.annotate(str(round(G2_det["error"][-1],3)), (G2_det["r"][-1], G2_det["error"][-1]))

        #     G_RJ_det = eval(df.loc["RWIS_G1", col])
        #     plt.plot(G_RJ_det["r"][skip:end], G_RJ_det["error"][skip:end], label = "RW-RJ with alpha: {}".format(alpha), marker = m_style)
        #     plt.annotate(str(round(G_RJ_det["error"][-1],3)), (G_RJ_det["r"][-1], G_RJ_det["error"][-1]))

        #     G_BB_det = eval(df.loc["RWIS_G2", col])
        #     plt.plot(G_BB_det["r"][skip:end], G_BB_det["error"][skip:end], label = "RW-BB with alpha: {}".format(alpha), marker = m_style)
        #     plt.annotate(str(round(G_BB_det["error"][-1],3)), (G_BB_det["r"][-1], G_BB_det["error"][-1]))

        # else:
        G1_det = eval(df.loc["RWIS_G1", col])
        plt.plot(G1_det["r"][skip:end], G1_det["error"][skip:end], label = "RW-IS on G", marker = m_style)
        # plt.annotate(str(round(G1_det["error"][-1],3)), (G1_det["r"][-1], G1_det["error"][-1]))

        G2_det = eval(df.loc["RWIS_G2", col])
        plt.plot(G2_det["r"][skip:end], G2_det["error"][skip:end], label = "RW-IS on "+r'$\mathcal{L}(\widehat{G})$', marker = m_style)
        # plt.annotate(str(round(G2_det["error"][-1],3)), (G2_det["r"][-1], G2_det["error"][-1]))

        G_RJ_det = eval(df.loc["RWRJ", col])
        plt.plot(G_RJ_det["r"][skip:end], G_RJ_det["error"][skip:end], label = "RW-RJ, "+r"$\alpha: {}$".format(alpha), marker = m_style)
        # plt.annotate(str(round(G_RJ_det["error"][-1],3)), (G_RJ_det["r"][-1], G_RJ_det["error"][-1]))

        G_BB_det = eval(df.loc["RWBB", col])
        if type(G_BB_det) is dict:
            plt.plot(G_BB_det["r"][skip:end], G_BB_det["error"][skip:end], label = "RW-BB, "+r"$\alpha: {}$".format(alpha), marker = m_style)
        # plt.annotate(str(round(G_BB_det["error"][-1],3)), (G_BB_det["r"][-1], G_BB_det["error"][-1]))

        xlim_min, xlim_max = plt.xlim(4000,40000)
        plt.xticks(ticks=list(np.arange(xlim_min, xlim_max+5000, 5000)), labels=np.arange(xlim_min/10000, xlim_max/10000+0.5, 0.5).round(2).tolist() )
        plt.grid(True)
        plt.legend(fontsize = 9,loc='upper right')
        plt.xlabel('Random Walk Steps '+ r"$(10^4)$")
        plt.ylabel('NRMSE')
        plt.title("{}".format(dataset_name))
        plt.tight_layout()
        plt.margins(0)
        plt.savefig("{}/{}_count_on_{}.png".format(save_loc, graphlet_name, dataset_name), bbox_inches='tight', pad_inches=0)

# ---------------------------------------------------------------------------------

skip = 3
def create_alpha_plots(csv_file:str, save_loc: str, RWBB= False):
    df_alpha_error = pd.read_csv(csv_file, index_col=0)
    # for each col, plot on a figure
    for col in df_alpha_error.columns:
        fig = plt.figure(figsize=(4,3))
        # plot for each alpha
        col_data = str(col).split("_")
        dataset_name = col_data[0]
        graphlet_name = col_data[1]
        n_skips = 0
        for alpha in df_alpha_error.index[1:]:
            if alpha < 0.5:
                continue
            if (len(str(df_alpha_error.loc[alpha, col]))) <= 4:
                n_skips +=1
                print("skipping {} on {}".format(graphlet_name, dataset_name))
                continue
            data = eval(df_alpha_error.loc[alpha, col])
            plt.plot(data["r"][skip:end], data["error"][skip:end], label = r"$\alpha={}$".format(alpha), marker = m_style)
        if n_skips < len(df_alpha_error.columns):
            xlim_min, xlim_max = plt.xlim(1000,9000)
            plt.xticks(ticks=list(np.arange(xlim_min, xlim_max+2000, 2000)), labels=np.arange(xlim_min/10000, xlim_max/10000+0.2, 0.2).round(2).tolist() )
            plt.grid(True)
            plt.legend(fontsize = 7,loc='upper right')
            plt.xlabel('Random Walk Steps '+ r"$(10^4)$")
            plt.ylabel('NRMSE')
            plt.title("{}".format(dataset_name))
            plt.tight_layout()
            plt.margins(0)
            if RWBB == False:
                plt.savefig("{}/alpha_test/alpha_{}_count_{}.png".format(save_loc, dataset_name, graphlet_name), 
                            bbox_inches='tight', pad_inches=0)
            else:
                plt.savefig("{}/alpha_test/alpha_BB_{}_count_{}.png".format(save_loc, dataset_name, graphlet_name), 
                            bbox_inches='tight', pad_inches=0)

# ---------------------------------------------------------------------------------

def create_error_bar(csv_file:str, dataset:list, graphlet_list: list[str], r_value:int, save_loc:str, num_fw = 4):
    for dataset_name in dataset:
        df = pd.read_csv(csv_file, index_col=0)
        graphlet_list.sort()
        X_axis = np.arange(len(graphlet_list))
        width = 0.15
        if num_fw%2==0:
            half_len = -1*(num_fw//2-0.5)
        else:
            half_len = -1*(num_fw//2)
        error_dict = {}
        f = plt.figure(figsize=(5.5,2))
        for fw in df.index:
            error_dict[fw] = {}
        for fw in df.index:
            for g in graphlet_list:
                data = eval(df.loc[fw, dataset_name+"_"+g])
                if type(data) is not dict:
                    continue
                # find the index for r_value
                idx = data["r"].index(r_value)
                error_dict[fw][g] = data["error"][idx]
            if len(error_dict[fw].values()) > 0:
                plt.bar(X_axis+(half_len*width), error_dict[fw].values(), width=width, label = fw_name_map[fw])
                half_len =half_len + 1
        plt.legend(loc="best", fontsize = 8.5)
        plt.grid(True)
        mapped_g_list = [g_name_map[i] for i in graphlet_list]
        plt.xticks(X_axis, mapped_g_list)
        plt.tight_layout()
        plt.savefig("{}/compare_bar/{}.png".format(save_loc, dataset_name), bbox_inches='tight', pad_inches=0.05)

# ---------------------------------------------------------------------------------

def create_alpha_error_bar(csv_file:str, dataset:list, graphlet_list: list[str], alpha_list: list[float], save_loc:str, technique: str):
    for dataset_name in dataset:
        df = pd.read_csv(csv_file, index_col=0)
        num_alpha = len(alpha_list)
        graphlet_list.sort()
        X_axis = np.arange(len(graphlet_list))
        width = 0.15
        if num_alpha%2==0:
            half_len = -1*(num_alpha//2-0.5)
        else:
            half_len = -1*(num_alpha//2)
        error_dict = {}
        f = plt.figure(figsize=(5.5,2))
        for alpha in df.index:
                error_dict[alpha] = {}
        for alpha in alpha_list:
            for g in graphlet_list:
                data = eval(df.loc[alpha, dataset_name+"_"+g])
                if type(data) is not dict:
                    continue
                # find the index for r_value
                # print(max(data["r"]))
                error_dict[alpha][g] = data["error"][-1]
            if len(error_dict[alpha].values()) > 0:
                plt.bar(X_axis+(half_len*width), error_dict[alpha].values(), width=width, label = r"$\alpha: {}$".format(alpha))
                half_len =half_len + 1
        plt.legend(loc="best", fontsize = 8.5)
        plt.grid(True)
        mapped_g_list = [g_name_map[i] for i in graphlet_list]
        plt.xticks(X_axis, mapped_g_list)
        plt.tight_layout()
        plt.ylabel("NRMSE")
        plt.savefig("{}/alpha_compare_bar/{}_{}.png".format(save_loc, dataset_name, technique), bbox_inches='tight', pad_inches=0.05)



# create_error_plot(error_detail_csv, "test_fresh/count_error_new")
# create_alpha_plots("dataset_alpha_BB_error_details.csv", "test_fresh", True)
# create_error_bar("dataset_error_details.csv", ["twitter-weak"], 
#                  ["cyclic-C3", "wedge", "leaf-C3", "butterfly"], 44000, "test_fresh",3)

create_alpha_error_bar("dataset_alpha_BB_error_details.csv", ["wiki-vote"], 
                 ["cyclic-C3", "wedge", "leaf-C3", "cyclic-C4", "clique-K4"], [0.5,0.6,0.7,0.8,0.9], "test_fresh", "RWBB")

create_alpha_error_bar("dataset_alpha_error_details.csv", ["wiki-vote"], 
                 ["cyclic-C3", "wedge", "leaf-C3", "cyclic-C4", "clique-K4", "butterfly"], [0.2,0.4,0.5,0.6,0.8,0.9], "test_fresh", "RWRJ")