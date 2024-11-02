#!/bin/bash
jupyter serverextension enable --py jupyter_http_over_ws

mkdir -p /tf/tensorflow-tutorials/keras
chmod -R a+rwx /tf/
mkdir /.local
chmod a+rwx /.local
apt-get update
apt-get install -y --no-install-recommends wget git

mkdir -p /tf/tensorflow-tutorials/understanding/images
cd /tf/tensorflow-tutorials/understanding/images
wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/understanding/images
wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/understanding/images/sngp.png
wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/understanding/sngp.ipynb

mkdir -p /tf/tensorflow-tutorials
cd /tf/tensorflow-tutorials/estimator
wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/estimator
wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/estimator/premade.ipynb
wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/estimator/linear.ipynb
wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/estimator/keras_model_to_estimator.ipynb

mkdir -p /tf/tensorflow-tutorials/images
cd /tf/tensorflow-tutorials/images

wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/images/cnn.ipynb
wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/images/index.md
wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/images/segmentation.ipynb
wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/images/data_augmentation.ipynb
wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/images/transfer_learning_with_hub.ipynb
wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/images/transfer_learning.ipynb
wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/images/classification.ipynb

mkdir -p /tf/tensorflow-tutorials/images/images
cd /tf/tensorflow-tutorials/images/images

wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/images/images/tensorboard_transfer_learning_with_hub.png
wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/images/images/fine_tuning.png
wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/images/images/before_fine_tuning.png

mkdir -p /tf/tensorflow-tutorials/load_data/
cd /tf/tensorflow-tutorials/load_data/

wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/load_data/images.ipynb
wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/load_data/video.ipynb
wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/load_data/pandas_dataframe.ipynb
wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/load_data/csv.ipynb
wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/load_data/tfrecord.ipynb
wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/load_data/text.ipynb
wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/load_data/numpy.ipynb

mkdir -p /tf/tensorflow-tutorials/load_data/images/csv
cd /tf/tensorflow-tutorials/load_data/images/csv

wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/load_data/images/csv/traffic.jpg
wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/load_data/images/csv/Titanic.jpg
wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/load_data/images/csv/fonts.jpg


mkdir -p /tf/tensorflow-tutorials/optimization
cd /tf/tensorflow-tutorials/optimization

wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/optimization/compression.ipynb

mkdir -p /tf/tensorflow-tutorials/distribute
cd /tf/tensorflow-tutorials/distribute

wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/distribute/save_and_load.ipynb
wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/distribute/multi_worker_with_ctl.ipynb
wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/distribute/keras.ipynb
wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/distribute/parameter_server_training.ipynb
wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/distribute/custom_training.ipynb
wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/distribute/input.ipynb
wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/distribute/multi_worker_with_keras.ipynb
wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/distribute/dtensor_keras_tutorial.ipynb
wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/distribute/multi_worker_with_estimator.ipynb
wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/distribute/dtensor_ml_tutorial.ipynb

mkdir -p /tf/tensorflow-tutorials/distribute/images
cd /tf/tensorflow-tutorials/distribute/images

wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/distribute/images/tensorboard_distributed_training_with_keras.png


mkdir -p /tf/tensorflow-tutorials/audio
cd /tf/tensorflow-tutorials/audio

wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/audio/music_generation.ipynb
wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/audio/transfer_learning_audio.ipynb
wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/audio/simple_audio.ipynb

mkdir -p /tf/tensorflow-tutorials/quickstart
cd /tf/tensorflow-tutorials/quickstart

wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/quickstart/beginner.ipynb
wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/quickstart/images
wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/quickstart/images/beginner
wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/quickstart/images/beginner/run_cell_icon.png
wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/quickstart/advanced.ipynb

mkdir -p /tf/tensorflow-tutorials/video
cd /tf/tensorflow-tutorials/video

wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/video/transfer_learning_with_movinet.ipynb
wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/video/video_classification.ipynb

mkdir -p /tf/tensorflow-tutorials/generative/
cd /tf/tensorflow-tutorials/generative/

wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/generative/adversarial_fgsm.ipynb
wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/generative/cvae.ipynb
wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/generative/data_compression.ipynb
wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/generative/pix2pix.ipynb
wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/generative/cyclegan.ipynb
wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/generative/dcgan.ipynb
wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/generative/autoencoder.ipynb
wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/generative/deepdream.ipynb
wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/generative/style_transfer.ipynb

mkdir -p /tf/tensorflow-tutorials/generative/images
cd /tf/tensorflow-tutorials/generative/images

wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/generative/images/horse2zebra_2.png
wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/generative/images/adversarial_example.png
wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/generative/images/cvae_latent_space.jpg
wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/generative/images/Green_Sea_Turtle_grazing_seagrass.jpg
wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/generative/images/stylized-image.png
wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/generative/images/dogception.png
wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/generative/images/kadinsky-turtle.png
wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/generative/images/intro_autoencoder_result.png
wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/generative/images/dis.png
wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/generative/images/gan_diagram.png
wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/generative/images/gan2.png
wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/generative/images/image_denoise_fmnist_results.png
wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/generative/images/gen.png
wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/generative/images/cyclegan_model.png
wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/generative/images/cycle_loss.png
wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/generative/images/kadinsky.jpg
wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/generative/images/horse2zebra_1.png
wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/generative/images/gan1.png

mkdir -p /tf/tensorflow-tutorials/interpretability
cd /tf/tensorflow-tutorials/interpretability

wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/interpretability/integrated_gradients.ipynb

mkdir -p /tf/tensorflow-tutorials/interpretability/images
cd /tf/tensorflow-tutorials/interpretability/images

wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/interpretability/images/IG_fireboat.png

mkdir -p /tf/tensorflow-tutorials/structured_data
cd /tf/tensorflow-tutorials/structured_data

wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/structured_data/images/raw_window.png
wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/structured_data/time_series.ipynb
wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/structured_data/feature_columns.ipynb
wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/structured_data/preprocessing_layers.ipynb
wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/structured_data/imbalanced_data.ipynb

mkdir -p /tf/tensorflow-tutorials/structured_data/images
cd /tf/tensorflow-tutorials/structured_data/images

wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/structured_data/images/multistep_autoregressive.png
wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/structured_data/images/multistep_last.png
wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/structured_data/images/conv_window.png
wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/structured_data/images/raw_window_1h.png
wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/structured_data/images/wide_window.png
wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/structured_data/images/lstm_1_window.png
wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/structured_data/images/residual.png
wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/structured_data/images/wide_conv_window.png
wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/structured_data/images/multistep_conv.png
wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/structured_data/images/multistep_repeat.png
wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/structured_data/images/raw_window_24h.png
wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/structured_data/images/baseline.png
wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/structured_data/images/split_window.png
wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/structured_data/images/multistep_dense.png
wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/structured_data/images/multistep_lstm.png
wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/structured_data/images/lstm_many_window.png
wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/structured_data/images/narrow_window.png
wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/structured_data/images/last_window.png

mkdir -p /tf/tensorflow-tutorials/reinforcement_learning
cd /tf/tensorflow-tutorials/reinforcement_learning

wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/reinforcement_learning/actor_critic.ipynb

mkdir -p /tf/tensorflow-tutorials/reinforcement_learning/images
cd /tf/tensorflow-tutorials/reinforcement_learning/images

wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/reinforcement_learning/images/cartpole-v0.gif


mkdir -p /tf/tensorflow-tutorials/tensorflow_text
cd /tf/tensorflow-tutorials/tensorflow_text

wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/tensorflow_text/README.md

mkdir -p /tf/tensorflow-tutorials/tensorflow_text/customization
cd /tf/tensorflow-tutorials/tensorflow_text/customization

wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/customization/custom_training_walkthrough.ipynb
wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/customization/custom_layers.ipynb

mkdir -p /tf/tensorflow-tutorials/tensorflow_text/customization/images
cd /tf/tensorflow-tutorials/tensorflow_text/customization/images

wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/customization/images/full_network_penguin.png
wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/customization/images/penguins_ds_species.png
wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/customization/basics.ipynb

mkdir -p /tf/tensorflow-tutorials/tensorflow_text/keras
cd /tf/tensorflow-tutorials/tensorflow_text/keras

wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/keras
wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/keras/save_and_load.ipynb
wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/keras/keras_tuner.ipynb
wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/keras/classification.ipynb
wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/keras/text_classification_with_hub.ipynb
wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/keras/overfit_and_underfit.ipynb
wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/keras/text_classification.ipynb
wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/keras/regression.ipynb

apt-get autoremove -y
apt-get remove -y wget

python3 -m ipykernel.kernelspec
