# %%

import io
import pandas as pd
from keras.models import Model
import wandb
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import tempfile
import os
from tensorflow.keras.layers import Embedding, LSTM


# set the max columns to display to avoid truncation
pd.set_option("display.max_columns", None)


def model_summary_to_df(model: Model) -> pd.DataFrame:
    """
    Convert a Keras model summary to a pandas DataFrame.

    Args:
        model: The Keras model to convert the summary of.

    Returns:
        A pandas DataFrame containing the model summary information.

    """
    # Capture the model summary
    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + "\n"))
    summary_string = stream.getvalue()
    stream.close()

    # Split summary into lines
    lines = summary_string.split("\n")

    # Dynamically find the header line and summary line index
    header_line_index, summary_line_index = 0, 0
    for i, line in enumerate(lines):
        if "Output Shape" in line and "Param #" in line:
            header_line_index = i
        if "Total params:" in line:
            summary_line_index = i

    # Use the found header line to determine column start positions
    header_line = lines[header_line_index]
    col_names = ["Layer", "Output Shape", "Param #"]
    col_starts = {col: header_line.find(col) for col in col_names}

    # Parse the model summary string
    model_lines = lines[header_line_index:summary_line_index]
    summary_lines = lines[-summary_line_index:]

    data = []
    # Parse layer information using column start positions
    for line in model_lines:
        if line == header_line or "===" in line:
            continue
        elif len(line.strip()) > 0:
            layer = line[: col_starts["Output Shape"]].strip()
            output_shape = line[
                col_starts["Output Shape"] : col_starts["Param #"]
            ].strip()
            params = line[col_starts["Param #"] :].strip()
            data.append([layer, output_shape, params])

    # Parse and add summary info
    for info in summary_lines:
        if info:  # Check if the line is not empty
            parts = info.split(":")
            if len(parts) == 2:
                info_label = parts[0].strip()
                info_value = parts[1].strip()
                data.append([info_label, "", info_value])

    df_tmp = pd.DataFrame(data, columns=col_names)

    # Merge rows that are split into two lines
    processed_rows = []
    for i in range(len(df_tmp)):
        row = df_tmp.iloc[i]
        # Check if the next row needs to be merged with the current row
        if (
            i + 1 < len(df_tmp)
            and df_tmp.iloc[i + 1]["Layer"].endswith(")")
            and df_tmp.iloc[i + 1]["Output Shape"].strip() == ""
            and df_tmp.iloc[i + 1]["Param #"].strip() == ""
        ):
            # Merge current and next row
            next_row = df_tmp.iloc[i + 1]
            merged_row = [
                row["Layer"] + next_row["Layer"],
                row["Output Shape"],
                row["Param #"],
            ]
            processed_rows.append(merged_row)
        elif (
            row["Layer"].endswith(")")
            and row["Output Shape"] == ""
            and row["Param #"] == ""
        ):
            continue
        else:
            processed_rows.append(row.values.tolist())

    return pd.DataFrame(processed_rows, columns=df_tmp.columns)


def wandb_log_model_summary_and_architecture(
    model: Model,
    log_summary: bool = True,
    log_architecture_plot: bool = False,
    prefix: str = "Model ",
    suffix: str = "",
) -> None:
    """
    Logs the model summary and architecture to wandb.

    Args:
        model: The model to log the summary and architecture for.
        log_summary: A boolean indicating whether to log the model summary. Default is True.
        log_architecture_plot: A boolean indicating whether to log the model architecture plot.

    Returns:
        None

    """
    if log_summary:
        wandb.log(
            {
                f"{prefix}Summary{suffix}": wandb.Table(
                    dataframe=model_summary_to_df(model)
                )
            }
        )
    if log_architecture_plot:
        # Use a temporary file to save the model plot
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".png", dir="."
        ) as tmpfile:
            tf.keras.utils.plot_model(
                model, to_file=tmpfile.name, show_shapes=True, show_layer_names=True
            )
            # Open the temporary file and log it to wandb
            wandb.log({f"{prefix}Architecture{suffix}": wandb.Image(tmpfile.name)})
        # Optionally, delete the temporary file if you don't want it to remain
        os.remove(tmpfile.name)


##%%

models = {
    "Basic_CNN": Sequential(
        [
            Conv2D(
                32,
                kernel_size=(3, 3),
                activation="relu",
                input_shape=(64, 64, 3),
                name="conv2d_1",
            ),
            MaxPooling2D(pool_size=(2, 2), name="max_pooling2d_1"),
            Conv2D(64, (3, 3), activation="relu", name="conv2d_2"),
            MaxPooling2D(pool_size=(2, 2), name="max_pooling2d_2"),
            Flatten(name="flatten"),
            Dense(128, activation="relu", name="dense_1"),
            Dropout(0.5, name="dropout"),
            Dense(10, activation="softmax", name="output"),
        ]
    ),
    "RNN_Text_Processing": Sequential(
        [
            Embedding(input_dim=10000, output_dim=128, name="embedding"),
            LSTM(64, return_sequences=True, name="lstm_1"),
            LSTM(32, name="lstm_2"),
            Dense(10, activation="softmax", name="output"),
        ]
    ),
    "Autoencoder": Sequential(
        [
            Dense(128, activation="relu", input_shape=(784,), name="dense_1"),
            Dense(64, activation="relu", name="dense_2"),
            Dense(128, activation="relu", name="dense_3"),
            Dense(784, activation="sigmoid", name="output"),
        ]
    ),
    "CNN_Regression": Sequential(
        [
            Conv2D(
                32,
                kernel_size=(3, 3),
                activation="relu",
                input_shape=(64, 64, 3),
                name="conv2d_1",
            ),
            MaxPooling2D(pool_size=(2, 2), name="max_pooling2d_1"),
            Flatten(name="flatten"),
            Dense(128, activation="relu", name="dense_1"),
            Dense(1, name="output"),  # Output layer for regression
        ]
    ),
    "Simple_MLP": Sequential(
        [
            Flatten(input_shape=(28, 28), name="flatten"),
            Dense(512, activation="relu", name="dense_1"),
            Dropout(0.2, name="dropout"),
            Dense(10, activation="softmax", name="output"),
        ]
    ),
}

# start a new run
wandb.init(project="dev", reinit=True)

for model_name, model in models.items():

    # print summary
    print(model.summary())

    # Get the model summary DataFrame
    df = model_summary_to_df(model)

    # log to wandb
    # assumes you have already done `wandb.init()`
    wandb_log_model_summary_and_architecture(
        model, log_summary=True, log_architecture_plot=True, prefix=f"{model_name} "
    )

    # Display the DataFrame
    print(df)

# %%

# %%
