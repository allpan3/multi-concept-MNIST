default:
	@echo "Please specify a target to make."

clean-samples:
	rm -rf tests/*/*/*samples.pt tests/*/*/*.json

clean-codebooks: clean-samples
	rm -rf tests/*/*/codebooks.pt

clean-all: clean-samples clean-codebooks
	find tests -type d -empty -delete