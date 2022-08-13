from utils import *
from DL_model import *
from jiwer import wer

def main():
    path_to_file = r'C:\Users\vikassaigiridhar\Music\spanish_translation\New folder\asr-spanish-v1-carlfm01\asr-spanish-v1-carlfm01\files.csv'
    data_path = r'C:\Users\vikassaigiridhar\Music\spanish_translation\New folder\asr-spanish-v1-carlfm01\asr-spanish-v1-carlfm01'
    lr = 0.001
    epochs = 50
    input_dim = (384 // 2) + 1
    output_dim = 35

    df = pd.read_csv(path_to_file)
    reduced_df = basic_eda(df, data_path)
    #print(reduced_df.head())
    train_df, test_df = train_test_split(reduced_df)
    train_ds, test_ds = tfto_dataset(train_df=train_df, test_df=test_df)
    model = audio_to_text_rnn(input_dim=input_dim, output_dim=output_dim, rnn_layers=5, rnn_units=128)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss=CTC_LOSS)
    history = model.fit(
        train_ds,
        epochs=epochs
    )

    predictions = []
    targets = []
    for batch in test_ds:
        X, y = batch
        batch_predictions = model.predict(X)
        batch_predictions = decode_batch_predictions(batch_predictions)
        predictions.extend(batch_predictions)
        for label in y:
            tf.strings.reduce_join(num_to_char(label).numpy().deocde("UTF-8"))
            targets.append(label)

    wer_score = wer(targets, predictions)
    print("=" * 100)
    print(f'Word Error Rate {wer_score}')
    print("=" * 100)

    for i in np.random.randint(0, len(predictions), 5):
        print(f"Target : {targets[i]}")
        print(f"Prediction : {predictions[i]}")
        print("+" * 100)

if __name__ == '__main__':
    main()


