

python -m sfw_brood.train \
  -n ${{ github.event.inputs.n_epochs }} -a ${{ github.event.inputs.cnn_arch }} \
  -b ${{ github.event.inputs.batch_size }} -l ${{ github.event.inputs.learning_rate }} \
  -d ${{ github.event.inputs.sample_duration }} -t ${{ github.event.inputs.target }} \
  -e feeding -c config/${{ github.event.inputs.data_config }} -w 12 \
  --samples-per-class ${{ github.event.inputs.samples_per_class }} \
  --age-mode ${{ github.event.inputs.age_mode }} \
  --age-range ${{ github.event.inputs.age_range }} \
  "$WORK_DIR/out/s${{ github.event.inputs.sample_duration }}" \
  "$WORK_DIR/s${{ github.event.inputs.sample_duration }}/audio"
