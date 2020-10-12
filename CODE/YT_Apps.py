import base64
import io
import os
import time
import pathlib

import pandas as pd
from dash.dependencies import Input, Output, State
import dash_table
import plotly.graph_objects as go
import plotly.express as px
import dash
import dash_core_components as dcc
import dash_html_components as html

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
external_stylesheets = ['css/bWLwgP.css']

"""
Parameters: 
    data files in data
Return:
    html object/charts
"""

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

pd.set_option('display.expand_frame_repr', False)

server = app.server

DATA_PATH = pathlib.Path(__file__).parent.joinpath("data_tsne").resolve()

def input_field(title, state_id, state_value, state_max, state_min):
    """Takes as parameter the title, state, default value and range of an input field, and output a Div object with
    the given specifications."""
    return html.Div(
        [
            html.P(
                title,
                style={
                    "display": "inline-block",
                    "verticalAlign": "mid",
                    "marginRight": "5px",
                    "margin-bottom": "0px",
                    "margin-top": "0px",
                },
            ),
            html.Div(
                [
                    dcc.Input(
                        id=state_id,
                        type="number",
                        value=state_value,
                        max=state_max,
                        min=state_min,
                        size=str(7),
                    )
                ],
                style={
                    "display": "inline-block",
                    "margin-top": "0px",
                    "margin-bottom": "0px",
                },
            ),
        ]
    )

# Generate the default scatter plot
tsne_df = pd.read_csv(DATA_PATH.joinpath("tsne_3d.csv"), index_col=0)

data = []

for idx, val in tsne_df.groupby(tsne_df.index):
    idx = int(idx)

    scatter = go.Scatter3d(
        name=f"Category_id {idx}",
        x=val["x"],
        y=val["y"],
        z=val["z"],
        mode="markers",
        marker=dict(size=2.5, symbol="circle"),
    )
    data.append(scatter)

# Layout for the t-SNE graph
tsne_layout = go.Layout(margin=dict(l=0, r=0, b=0, t=0))

local_layout = html.Div(
    [
        # In-browser storage of global variables
        html.Div(id="data-df-and-message", style={"display": "none"}),
        html.Div(id="label-df-and-message", style={"display": "none"}),
        # Main app
        html.Div(
            [
                html.H3(
                    "tsne visualization of Youtube videos ",
                    id="title",
                    style={
                        "float": "left",
                        "margin-top": "20px",
                        "margin-bottom": "0",
                        "margin-left": "7px",
                    },
                ),
                # html.Img(
                #     src="",
                #     style={"height": "100px", "float": "right"},
                # ),
            ],
            className="row",
        ),
        html.Div(
            [
                html.Div(
                    [
                        # Data about the graph
                        html.Div(id="kl-divergence", style={"display": "none"}),
                        html.Div(id="end-time", style={"display": "none"}),
                        html.Div(id="error-message", style={"display": "none"}),
                        # The graph
                        dcc.Graph(
                            id="tsne-3d-plot",
                            figure={"data": data, "layout": tsne_layout},
                            style={"height": "80vh"},
                        ),
                    ],
                    id="plot-div",
                    className="eight columns",
                ),
                html.Div(
                    [
                        html.H4("t-SNE Parameters", id="tsne_h4"),
                        input_field("Perplexity:", "perplexity-state", 20, 50, 5),
                        input_field(
                            "Number of Iterations:", "n-iter-state", 400, 1000, 250
                        ),
                        input_field("Learning Rate:", "lr-state", 200, 1000, 10),
                        input_field(
                            "Initial PCA dimensions:", "pca-state", 3, 5, 3
                        ),
                        html.Button(
                            id="tsne-train-button",
                            n_clicks=0,
                            children="Start Training t-SNE",
                        ),
                        dcc.Upload(
                            id="upload-data",
                            children=html.A("Upload your input data here."),
                            style={
                                "height": "45px",
                                "line-height": "45px",
                                "border-width": "1px",
                                "border-style": "dashed",
                                "border-radius": "5px",
                                "text-align": "center",
                                "margin-top": "5px",
                                "margin-bottom": "5 px",
                            },
                            multiple=False,
                            max_size=-1,
                        ),
                        dcc.Upload(
                            id="upload-label",
                            children=html.A("Upload your labels here."),
                            style={
                                "height": "45px",
                                "line-height": "45px",
                                "border-width": "1px",
                                "border-style": "dashed",
                                "border-radius": "5px",
                                "text-align": "center",
                                "margin-top": "5px",
                                "margin-bottom": "5px",
                            },
                            multiple=False,
                            max_size=-1,
                        ),
                        html.Div(
                            [
                                html.P(
                                    id="upload-data-message",
                                    style={"margin-bottom": "0px"},
                                ),
                                html.P(
                                    id="upload-label-message",
                                    style={"margin-bottom": "0px"},
                                ),
                                html.Div(
                                    id="training-status-message",
                                    style={"margin-bottom": "0px", "margin-top": "0px"},
                                ),
                                html.P(id="error-status-message"),
                            ],
                            id="output-messages",
                            style={"margin-bottom": "2px", "margin-top": "2px"},
                        ),
                    ],
                    className="four columns",
                    style={
                        "padding": 20,
                        "margin": 5,
                        "borderRadius": 5,
                        "border": "thin lightgrey solid",
                        # Remove possibility to select the text for better UX
                        "user-select": "none",
                        "-moz-user-select": "none",
                        "-webkit-user-select": "none",
                        "-ms-user-select": "none",
                    },
                ),
            ],
            className="row",
        ),
        html.Div(
            [
                dcc.Markdown(
                    """
Youtube trending data mining using t-sne algorithm and plotly: https://dash.plotly.com/
"""
                )
            ],
            style={"margin-top": "15px"},
            className="row",
        ),
    ],
    className="container",
    style={"width": "90%", "max-width": "none", "font-size": "1.5rem"},
)

def local_callbacks(app):
    def parse_content(contents, filename):
        """This function parses the raw content and the file names, and returns the dataframe containing the data, as well
        as the message displaying whether it was successfully parsed or not."""

        if contents is None:
            return None, ""

        content_type, content_string = contents.split(",")

        decoded = base64.b64decode(content_string)

        try:
            if "csv" in filename:
                # Assume that the user uploaded a CSV file
                df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
            elif "xls" in filename:
                # Assume that the user uploaded an excel file
                df = pd.read_excel(io.BytesIO(decoded))

            else:
                return None, "The file uploaded is invalid."
        except Exception as e:
            print(e)
            return None, "There was an error processing this file."

        return df, f"{filename} successfully processed."

    # Uploaded data --> Hidden Data Div
    @app.callback(
        Output("data-df-and-message", "children"),
        [Input("upload-data", "contents"), Input("upload-data", "filename")],
    )
    def parse_data(contents, filename):
        data_df, message = parse_content(contents, filename)

        if data_df is None:
            return [None, message]

        elif data_df.shape[1] < 3:
            return [None, f"The dimensions of {filename} are invalid."]

        return [data_df.to_json(orient="split"), message]

    # Uploaded labels --> Hidden Label div
    @app.callback(
        Output("label-df-and-message", "children"),
        [Input("upload-label", "contents"), Input("upload-label", "filename")],
    )
    def parse_label(contents, filename):
        label_df, message = parse_content(contents, filename)

        if label_df is None:
            return [None, message]

        elif label_df.shape[1] != 1:
            return [None, f"The dimensions of {filename} are invalid."]

        return [label_df.to_json(orient="split"), message]

    # Hidden Data Div --> Display upload status message (Data)
    @app.callback(
        Output("upload-data-message", "children"),
        [Input("data-df-and-message", "children")],
    )
    def output_upload_status_data(data):
        return data[1]

    # Hidden Label Div --> Display upload status message (Labels)
    @app.callback(
        Output("upload-label-message", "children"),
        [Input("label-df-and-message", "children")],
    )
    def output_upload_status_label(data):
        return data[1]

    # Button Click --> Update graph with states
    @app.callback(
        Output("plot-div", "children"),
        [Input("tsne-train-button", "n_clicks")],
        [
            State("perplexity-state", "value"),
            State("n-iter-state", "value"),
            State("lr-state", "value"),
            State("pca-state", "value"),
            State("data-df-and-message", "children"),
            State("label-df-and-message", "children"),
        ],
    )
    def update_graph(
        n_clicks, perplexity, n_iter, learning_rate, pca_dim, data_div, label_div
    ):
        """Run the t-SNE algorithm upon clicking the training button"""

        error_message = None  # No error message at the beginning

        # Fix for startup POST
        if n_clicks <= 0 or data_div is None or label_div is None:
            global data
            kl_divergence, end_time = None, None

        else:
            # Extract the data dataframe and the labels dataframe from the divs. they are both the first child of the div,
            # and are serialized in json
            data_df = pd.read_json(data_div[0], orient="split")
            label_df = pd.read_json(label_div[0], orient="split")

            # Fix the range of possible values
            if n_iter > 1000:
                n_iter = 1000
            elif n_iter < 250:
                n_iter = 250

            if perplexity > 50:
                perplexity = 50
            elif perplexity < 5:
                perplexity = 5

            if learning_rate > 1000:
                learning_rate = 1000
            elif learning_rate < 10:
                learning_rate = 10

            # We limit the pca_dim to the dimensionality of the dataset
            if pca_dim > data_df.shape[1]:
                pca_dim = data_df.shape[1]
            elif pca_dim < 3:
                pca_dim = 3

            # Start timer
            start_time = time.time()

            # Apply PCA on the data first
            pca = PCA(n_components=pca_dim)
            data_pca = pca.fit_transform(data_df)

            # Then, apply t-SNE with the input parameters
            tsne = TSNE(
                n_components=3,
                perplexity=perplexity,
                learning_rate=learning_rate,
                n_iter=n_iter,
            )

            try:
                data_tsne = tsne.fit_transform(data_pca)
                kl_divergence = tsne.kl_divergence_

                # Combine the reduced t-sne data with its label
                tsne_data_df = pd.DataFrame(data_tsne, columns=["x", "y", "z"])

                label_df.columns = ["label"]

                combined_df = tsne_data_df.join(label_df)

                data = []

                # Group by the values of the label
                for idx, val in combined_df.groupby("label"):
                    scatter = go.Scatter3d(
                        name=idx,
                        x=val["x"],
                        y=val["y"],
                        z=val["z"],
                        mode="markers",
                        marker=dict(size=2.5, symbol="circle"),
                    )
                    data.append(scatter)

                end_time = time.time() - start_time

            # Catches Heroku server timeout
            except:
                error_message = "We were unable to train the t-SNE model due to timeout. Try to run it again, or to clone the repo and run the program locally."
                kl_divergence, end_time = None, None

        return [
            # Data about the graph
            html.Div([kl_divergence], id="kl-divergence", style={"display": "none"}),
            html.Div([end_time], id="end-time", style={"display": "none"}),
            html.Div([error_message], id="error-message", style={"display": "none"}),
            # The graph
            dcc.Graph(
                id="tsne-3d-plot",
                figure={"data": data, "layout": tsne_layout},
                style={"height": "80vh"},
            ),
        ]

    # Updated graph --> Training status message
    @app.callback(
        Output("training-status-message", "children"),
        [Input("end-time", "children"), Input("kl-divergence", "children")],
    )
    def update_training_info(end_time, kl_divergence):
        # If an error message was output during the training.

        if (
            end_time is None
            or kl_divergence is None
            or end_time[0] is None
            or kl_divergence[0] is None
        ):
            return None
        else:
            end_time = end_time[0]
            kl_divergence = kl_divergence[0]

            return [
                html.P(
                    f"t-SNE trained in {end_time:.2f} seconds.",
                    style={"margin-bottom": "0px"},
                ),
                html.P(
                    f"Final KL-Divergence: {kl_divergence:.2f}",
                    style={"margin-bottom": "0px"},
                ),
            ]

    @app.callback(
        Output("error-status-message", "children"), [Input("error-message", "children")]
    )
    def show_error_message(error_message):
        if error_message is not None:
            return [html.P(error_message[0])]

        else:
            return []


#---------------------------------------------------------------------
# import trending dates data
df= pd.read_csv('data/trending_dates_among_countries.csv',header=None)
# extract total unique categories from csv files
categories = df.iloc[0].values.tolist()
categories = categories[1:]
# trending days  = trending_date - publish_date
US_trending = df.iloc[1].values.tolist()
US_trending= US_trending[1:]
GB_trending = df.iloc[2].values.tolist()
GB_trending = GB_trending[1:]
CA_trending = df.iloc[3].values.tolist()
CA_trending = CA_trending[1:]

#---------------------------------------------------------------------
# import tags
us_tags = pd.read_csv('data/us_tags.csv')
CA_tags = pd.read_csv('data/CA_tags.csv')
GB_tags = pd.read_csv('data/GB_tags.csv')

us_hist = px.histogram(us_tags, x="number of tags",
                        color="category_id", 
                        title='US: videos tags distribution across all categories')
CA_hist = px.histogram(CA_tags, x="number of tags",
                        color="category_id",
                        title='Canada: videos tags distribution across all categories')
GB_hist = px.histogram(GB_tags, x="number of tags",
                        color="category_id",
                        title='Great Britain: videos tags distribution across all categories')
#---------------------------------------------------------------------
# import linear regression results
dftb = pd.read_csv('data/USvideo_edited_partial1table.csv')
df = pd.read_csv('data/USvideo_edited_partial1.csv')
dftb['id'] = dftb['video_id']
dftb.set_index('id', inplace=True, drop=False)

#---------------------------------------------------------------------
# app layout including all interactive plots
app.layout=html.Div(children = [
    # title
    html.Div([
            html.H2(children='Interactive Visualization of YouTube Trending Videos: Data Mining & Modeling')
            ], style={'marginLeft': 100}),

    # tsne page
    local_layout,

    # interactive table
    html.Div([
        #html.H1(children='YouTube Trending videos'),
        html.H2(children='Basic information of popular YouTube videos'),
        html.H3(children='Sort, Filter and More for United States dataset'),

        dash_table.DataTable(
            id='datatable-row-ids',
            columns=[
                {'name': i, 'id': i, 'deletable': True} for i in dftb.columns
                # omit the id column
                if i != 'id'
            ],
            data=dftb.to_dict('records'),
            editable=True,
            filter_action="native",
            sort_action="native",
            sort_mode='multi',
            row_selectable='multi',
            row_deletable=True,
            selected_rows=[],
            page_action='native',
            page_current= 0,
            page_size= 10,
            style_cell={'fontSize':15, 'font-family':'sans-serif'},
            style_header = {'fontSize':15, 'font-family':'sans-serif'},
        ),
        html.Div(id='datatable-row-ids-container')
    ]),

    # stacked bar plot
    dcc.Graph(
        id='stacked-graph',
        figure={
            'data': [
                {'x': categories, 'y': US_trending, 'type': 'bar', 'name': 'United States'},
                {'x': categories, 'y': CA_trending, 'type': 'bar', 'name': 'Canada'},
                {'x': categories, 'y': GB_trending, 'type': 'bar', 'name': 'Great British'},
                ],
            'layout': {
                'title': 'Trending dates of Different Categories among Different Countries',
                'xaxis':{
                    'title':'categories'},
                'yaxis':{
                   'title':'Trending Dates (days)'},
                }
            }
    ),

    # histogram plot
    dcc.Graph(
            id='us_hist',
            figure=us_hist),
    dcc.Graph(
            id='CA_hist',
            figure=CA_hist),
    dcc.Graph(
            id='GB_hist',
            figure=GB_hist),

    # linear regression
    html.Div([

        dcc.Graph(
            id='linear-regression',
            figure={
                'data': [
                    go.Scatter(
                        x=df[df['category'] == i]['Likes_Actual'],
                        y=df[df['category'] == i]['Likes_Predicted'],
                        text=df[df['category'] == i]['video_id'],
                        mode='markers',
                        opacity=0.8,
                        marker={
                            'size': 13,
                            'line': {'width': 0.5, 'color': 'white'}
                        },
                        name=i
                    ) for i in df.category.unique()
                ],
                'layout': go.Layout(
                    xaxis = dict(title_text = "Predicted number of likes",
                                title_font = {"size": 20},
                                title_standoff = 100),
                    yaxis = dict(title_text = "Actual number of likes",
                                title_font = {"size": 20},
                                scaleanchor = "x",
                                scaleratio = 1,
                                title_standoff = 100),
                    
                    font=dict(
                        family="Courier New, monospace",
                        size=18,
                        color="#000000"
                    ),
                    margin={'l': 40, 'b': 40, 't': 40, 'r': 40},
                    legend_title='<b> Video category </b>',
                    autosize=False,
                    width=1000,
                    height=650,
                    hovermode='closest'
                )
            }
        )
    ], style={'marginLeft': 300})
    ])

#---------------------------------------------------------------------
@app.callback(
    Output('datatable-row-ids-container', 'children'),
    [Input('datatable-row-ids', 'derived_virtual_row_ids'),
     Input('datatable-row-ids', 'selected_row_ids'),
     Input('datatable-row-ids', 'active_cell')])


def update_graphs(row_ids, selected_row_ids, active_cell):
    # When the table is first rendered, `derived_virtual_data` and
    # `derived_virtual_selected_rows` will be `None`. This is due to an
    # idiosyncracy in Dash (unsupplied properties are always None and Dash
    # calls the dependent callbacks when the component is first rendered).
    # So, if `rows` is `None`, then the component was just rendered
    # and its value will be the same as the component's dataframe.
    # Instead of setting `None` in here, you could also set
    # `derived_virtual_data=df.to_rows('dict')` when you initialize
    # the component.
    selected_id_set = set(selected_row_ids or [])

    if row_ids is None:
        dfftb = dftb
        # pandas Series works enough like a list for this to be OK
        row_ids = dftb['id']
    else:
        dfftb = dftb.loc[row_ids]

    active_row_id = active_cell['row_id'] if active_cell else None

    colors = ['#FF69B4' if id == active_row_id
              else '#7FDBFF' if id in selected_id_set
              else '#0074D9'
              for id in row_ids]

    return [
        dcc.Graph(
            id=column + '--row-ids',
            figure={
                'data': [
                    {
                        'x': dfftb['video_id'],
                        'y': dfftb[column],
                        'type': 'bar',
                        'marker': {'color': colors},
                    }
                ],
                'layout': {
                    'xaxis': {'automargin': True},
                    'yaxis': {
                        'automargin': True,
                        'title': {'text': column}
                    },
                    'height': 250,
                    'margin': {'t': 10, 'l': 10, 'r': 10},
                },
            },
        )
        # check if column exists - user may have deleted it
        # If `column.deletable=False`, then you don't
        # need to do this check.
        for column in ['Likes', 'views', 'comment_count', 'dislikes'] if column in dfftb
    ]

# callbacks of the tsne App
local_callbacks(app)

if __name__ == '__main__':
    app.run_server(debug=True)