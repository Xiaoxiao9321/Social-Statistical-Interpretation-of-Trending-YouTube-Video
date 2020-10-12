1. DESCRIPTION
The visualization of this application is based on Python, plotly, plotly Dash and flask.  Precessed data files and modeling results used in this application is placed under folder "data". css is adopted for setting of html elements implmented in this application.  User doesn't need to config the flask web server as it is configued through the Dash package.

Prerequisite:
1) Python >= 3.7.6
2) All necessary packages are listed below:
dash==1.11.0
gunicorn==19.9.0
numpy==1.16.3
pandas==0.24.2
scikit-learn==0.21.0
scipy==1.2.1
pillow==6.2.1
plotly==4.6.0

2. INSTALLATION
1)  install dash
pip install dash==1.11.0
Note: starting with dash 0.37.0, dash automatically installs dash-renderer, dash-core-components, dash-html-components, and dash-table, using known-compatible versions of each. You need not and should not install these separately any longer, only dash itself.

2) install Python-specific Requirements: Virtualenv
cd to the directory where requirements.txt is placed (in the CODE folder).
pip install -r requirements.txt

3. EXECUTION
1) cd to the CODE folder, and run the following command in powershell/cmd(windows) or terminal(Linux,Mac)
python YT_Apps.py

2) open the link in your browser,  http://127.0.0.1:8050/

3) the datsets for tsne is placed under CODE/data_tsne.  User is can upload eg. CA_data.csv and CA_label.csv into the Application by click corresponding button/label in the tsne section.  It is suggested to use the default training parameter. Once the data successfully processed, click the train_model button to run the tsne modeling training.  It will take ~5 mins to run one training process.

