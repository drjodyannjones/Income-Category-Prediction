# Income Classification Prediction

In this project, I developed a classification model on publicly available Census Bureau data. I created unit tests in order to monitor the model's performance on various data slices. I deployed the model using the FastAPI package and created the appropriate API tests. The slice validation and the API tests will be incorporated into a CI/CD framework using GitHub Actions.

If you would like to replicate this project, here are the steps:

## Step 1: Clone this repo

Clone this project into a directory of your choice. For example, mine is saved in a subdirectory Documents/Projects. Give the project a relevant name so you can remember what it is about at a later date. You can code this repo by typing:

<pre>git clone https://github.com/drjodyannjones/IncomeCategoryPrediction.git</pre>

## Step 2: Crete a project directory

Decide on and create a project directory structure. This project uses the following directory structure. Feel free to replicate it or create your own:

<pre>
project_name/
│
├── data/                       # Data folder
│   ├── raw/                    # Raw data files
│   │   └── census.csv          # CSV file containing the raw dataset
│   └── processed/              # Processed data files
│
├── notebooks/                  # Jupyter notebooks for development, exploration, and visualization
│
├── scripts/                    # Utility scripts for data preprocessing or model evaluation
│
├── src/                        # Main source code folder
│   ├── app/                    # Main application folder
│   │   ├── api/                # API endpoints and routes
│   │   │   ├── __init__.py
│   │   │   └── endpoints.py
│   │   ├── models/             # ML model-related code and files
│   │   │   ├── __init__.py
│   │   │   ├── data.py
│   │   │   ├── train_model.py
│   │   │   └── model.pkl       # Serialized model file
│   │   ├── utils/              # Utility functions and classes
│   │   │   ├── __init__.py
│   │   │   └── utils.py
│   │   ├── __init__.py
│   │   ├── config.py           # Configuration file
│   │   └── main.py             # FastAPI app initialization and configuration
│   └── __init__.py
│
├── tests/                      # Test cases and files
│   ├── __init__.py
│   └── test_app.py
│
├── Dockerfile                  # Docker configuration file
├── requirements.txt            # Project dependencies
├── setup.py                    # Package and distribution setup
├── .env.example                # Example environment configuration file
├── .gitignore                  # Git ignore file
└── README.md                   # Project documentation
</pre>

## Step 3: Setup virtual (conda) environment AND Activate it

Once you have your project directory setup, create a virtual environment. I use conda. Here are the instructions to do that:

<pre>conda create -n [envname] "python=3.8" scikit-learn pandas numpy pytest jupyter jupyterlab fastapi uvicorn -c conda-forge</pre>

Remember to replace [envname] with the respective name of your virtual environment. Name it something that is relevant to the project e.g. income_classification.

In order to activate your virtual environment, run this command:

<pre>conda activate [envname]</pre>

## Step 4: Install the dependencies into your virtual environment

<pre>conda install --file requirements.txt</pre>
