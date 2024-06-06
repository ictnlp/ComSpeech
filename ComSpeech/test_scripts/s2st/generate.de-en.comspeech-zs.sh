exp=s2st.de-en.comspeech-zs

ROOT=~/speech2speech
output_dir=ComSpeech/results/$exp
checkpoint_dir=ComSpeech/checkpoints/$exp

python fairseq/scripts/average_checkpoints.py \
    --inputs $checkpoint_dir/checkpoint.best_loss*.pt \
    --output $checkpoint_dir/average_best_checkpoint.pt

python ComSpeech/generator/generate_features.py \
    data/comspeech/cvss_de_en/s2s \
    --user-dir ComSpeech \
    --config-yaml config.yaml --gen-subset test.full --task comspeech_task \
    --validate-task s2st --max-tokens 40000 \
    --path $checkpoint_dir/average_best_checkpoint.pt \
    --required-batch-size-multiple 1 \
    --beam 10 --lenpen 1.0 \
    --results-path $output_dir

python hifi-gan/inference_e2e.py \
    --input_mels_dir $output_dir/feat \
    --output_dir $output_dir/wav \
    --checkpoint_file hifi-gan/VCTK_V1/generator_v1

cd $output_dir/wav/
for file in *generated_e2e.wav; do
    newname=$(echo "$file" | sed 's/generated_e2e\.wav$/pred.wav/')
    mv "$file" "$newname"
done

cd $ROOT/asr_bleu/
python compute_asr_bleu.py \
  --lang en \
  --audio_dirpath $ROOT/$output_dir/wav \
  --reference_path $ROOT/data/comspeech/cvss_de_en/s2s/test.txt \
  --reference_format txt