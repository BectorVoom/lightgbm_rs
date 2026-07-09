while true; do
  STATUS=$(kaggle kernels status boomvector/lgb-rs-cuda-bench 2>&1)
  echo "$STATUS"
  if [[ "$STATUS" == *"COMPLETE"* || "$STATUS" == *"ERROR"* || "$STATUS" == *"CANCELLED"* ]]; then
    break
  fi
  sleep 60
done
kaggle kernels output boomvector/lgb-rs-cuda-bench -p kaggle_out
cat kaggle_out/*
