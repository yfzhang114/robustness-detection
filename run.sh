# set the dataset pair that will be used
data='xsum'
data2='squad'


# set the machine

# model='meta-llama/Llama-2-7b-hf'
# model='lmsys/vicuna-v1.5'
# model='/n/holylabs/LABS/barak_lab/Lab/models/llama_models/7B'
# model='/n/holylabs/LABS/barak_lab/Lab/models/llama_models/alpaca-7B'
model='/n/holylabs/LABS/barak_lab/Lab/models/llama_models/vicuna-7b-v1.5'

# echo "runing test"
echo $data
echo $data2
echo $model


# evaluating all the zero-shot detectors with the same Human-authored sources and machine generated texts
python run_ood.py --output_name main --base_model_name $model --mask_filling_model_name t5-3b --baselines_only --n_perturbation_list 1,10 --n_samples 500 --pct_words_masked 0.3 --span_length 2 --dataset $data --dataset2 $data >  vicuna-7b-baselines-t5-3b-$data-$data.out
python run_ood.py --output_name main --base_model_name $model --mask_filling_model_name t5-3b --baselines_only --n_perturbation_list 1,10 --n_samples 500 --pct_words_masked 0.3 --span_length 2 --dataset $data2 --dataset2 $data2 >  vicuna-7b-baselines-t5-3b-$data2-$data2.out


# evaluating all the zero-shot detectors with different Human-authored sources and machine generated texts
echo "transfer from data 1 to all"
python run_ood.py --output_name main --base_model_name $model --mask_filling_model_name t5-3b --baselines_only --n_perturbation_list 1,10 --n_samples 500 --pct_words_masked 0.3 --span_length 2 --dataset $data --dataset2 'squad' >  vicuna-7b-baselines-t5-3b-$data-squad.out
python run_ood.py --output_name main --base_model_name $model --mask_filling_model_name t5-3b --baselines_only --n_perturbation_list 1,10 --n_samples 500 --pct_words_masked 0.3 --span_length 2 --dataset $data --dataset2 'code' >  vicuna-7b-baselines-t5-3b-$data-code.out
python run_ood.py --output_name main --base_model_name $model --mask_filling_model_name t5-3b --baselines_only --n_perturbation_list 1,10 --n_samples 500 --pct_words_masked 0.3 --span_length 2 --dataset $data --dataset2 'xsum' >  vicuna-7b-baselines-t5-3b-$data-xsum.out
python run_ood.py --output_name main --base_model_name $model --mask_filling_model_name t5-3b --baselines_only --n_perturbation_list 1,10 --n_samples 500 --pct_words_masked 0.3 --span_length 2 --dataset $data --dataset2 'writing' >  vicuna-7b-baselines-t5-3b-$data-writing.out

echo "transfer from other datasets to data 1"
python run_ood.py --output_name main --base_model_name $model --mask_filling_model_name t5-3b --baselines_only --n_perturbation_list 1,10 --n_samples 500 --pct_words_masked 0.3 --span_length 2 --dataset $data --dataset2 'squad' >  vicuna-7b-baselines-t5-3b-$data-squad.out
python run_ood.py --output_name main --base_model_name $model --mask_filling_model_name t5-3b --baselines_only --n_perturbation_list 1,10 --n_samples 500 --pct_words_masked 0.3 --span_length 2 --dataset $data --dataset2 'code' >  vicuna-7b-baselines-t5-3b-$data-code.out
python run_ood.py --output_name main --base_model_name $model --mask_filling_model_name t5-3b --baselines_only --n_perturbation_list 1,10 --n_samples 500 --pct_words_masked 0.3 --span_length 2 --dataset $data --dataset2 'xsum' >  vicuna-7b-baselines-t5-3b-$data-xsum.out
python run_ood.py --output_name main --base_model_name $model --mask_filling_model_name t5-3b --baselines_only --n_perturbation_list 1,10 --n_samples 500 --pct_words_masked 0.3 --span_length 2 --dataset $data --dataset2 'writing' >  vicuna-7b-baselines-t5-3b-$data-writing.out
