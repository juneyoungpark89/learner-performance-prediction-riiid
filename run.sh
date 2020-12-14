echo "LINTER CHECK"
echo "----------------------------"
echo "BLACK"
black . --config=.github/linters/.python-black
echo "----------------------------"
echo "Flake8"
flake8 . --config=.github/linters/.flake8
echo "----------------------------"
