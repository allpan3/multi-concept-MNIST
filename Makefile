default:
	@echo "Please specify a target to make."

clean-checkpoints:
	rm -rf data/*/*.checkpoint

clean-samples:
	rm -rf data/*/*samples.pt data/*/*.json

clean-codebooks: clean-samples
	rm -rf data/*/codebooks.pt