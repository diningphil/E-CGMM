To launch the experiment:

python launch_experiment.py --model [ECGMM or CGMM] --dataset-name [pubmed cora or citeseer] --data-root [your data folder] --trials-per-conf [number of trial for each configuration] --config-file [your configuration(s) for the E-CGMM model] --MLP-config-file [your configuration(s) for the MLP] --result-folder [your results folder] --final-training-runs [number of final training runs] 

NB: Remember to change 'dim_node_features' inside model_config for the specific dataset you are testing.
pubmed   ---> 3
cora     ---> 7
citeseer ---> 6
