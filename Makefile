default:
	@echo "Please specify a target to make."

clean-samples:
	rm -rf tests/*/*/*samples.pt

clean-codebooks: clean-samples
	rm -rf tests/*/*/codebooks.pt