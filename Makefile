install:
	@pip install -e .

clean:
	@rm -f */version.txt
	@rm -f .coverage
	@rm -rf */.ipynb_checkpoints
	@rm -Rf build
	@rm -Rf */__pycache__
	@rm -Rf */*.pyc
	@echo "🧽 Cleaned up successfully!"

all: install clean

test:
	@pytest -v tests

app:
	@streamlit run ai_medreview/app.py

app3:
	@streamlit run ai_medreview/app_test.py

app_wide:
	@streamlit run ai_medreview/app_wide.py

gsheet:
	@streamlit run ai_medreview/gsheet_connect.py

data:
	@python ai_medreview/data.py

repeat:
	@python ai_medreview/scheduler.py

git_merge:
	$(MAKE) clean
	@python ai_medreview/auto_git/git_merge.py

git_push:
	$(MAKE) clean
	$(MAKE) lint
	@python ai_medreview/auto_git/git_push.py


lint:
	@black ai_medreview/

