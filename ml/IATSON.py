import keras
from keras.layers import LeakyReLU


class IATSON:
    def __init__(self) -> None:
        super().__init__()

    def get_flux_conv_branch(self, name):
        # (time, flux, detrended_flux1 ... detrended_flux5, flux_model, centroidx, centroidy, motionx, motiony, bck)
        # flux model by transit params and stellar params
        leaky_relu_alpha = 0.01
        flux_input = keras.Input(shape=(2500, 13), name=name)
        flux_branch = keras.layers.SpatialDropout1D(rate=0.2)(flux_input)
        flux_branch = keras.layers.Conv1D(filters=128, kernel_size=9, padding="same",
                                          activation=LeakyReLU(leaky_relu_alpha))(flux_branch)
        flux_branch = keras.layers.MaxPooling1D(pool_size=10, strides=6)(flux_branch)
        flux_branch = keras.layers.Dropout(rate=0.1)(flux_branch)
        flux_branch = keras.layers.Conv1D(filters=64, kernel_size=7, padding="same",
                                          activation=LeakyReLU(leaky_relu_alpha))(flux_branch)
        flux_branch = keras.layers.MaxPooling1D(pool_size=4, strides=3)(flux_branch)
        flux_branch = keras.layers.Conv1D(filters=32, kernel_size=5, padding="same",
                                          activation=LeakyReLU(leaky_relu_alpha))(flux_branch)
        flux_branch = keras.layers.MaxPooling1D(pool_size=3, strides=3)(flux_branch)
        flux_branch = keras.layers.Conv1D(filters=16, kernel_size=3, padding="same",
                                          activation=LeakyReLU(leaky_relu_alpha))(flux_branch)
        flux_branch = keras.layers.MaxPooling1D(pool_size=2, strides=2)(flux_branch)
        return flux_input, flux_branch

    def get_flux_model_branch(self):
        leaky_relu_alpha = 0.01
        flux_input, flux_branch = self.get_flux_conv_branch("global_flux_branch")
        flux_branch = keras.layers.Dense(16, activation=LeakyReLU(leaky_relu_alpha))(flux_branch)
        flux_branch = keras.layers.Dropout(rate=0.1)(flux_branch)
        flux_branch = keras.layers.Dense(16, activation='linear')(flux_branch)
        flux_branch = keras.layers.LayerNormalization()(flux_branch)
        flux_branch = keras.Model(inputs=[flux_input], outputs=flux_branch)
        return flux_branch

    def get_focus_flux_branch(self, name):
        leaky_relu_alpha = 0.01
        focus_flux_input = keras.Input(shape=(500, 13), name=name)
        # (time, flux, detrended_flux1... detrended_flux5, flux_model, centroidx, centroidy, motionx, motiony, bck)
        # flux model by transit params and stellar params
        focus_flux_branch = keras.layers.SpatialDropout1D(rate=0.2)(focus_flux_input)
        focus_flux_branch = keras.layers.Conv1D(filters=64, kernel_size=5, padding="same",
                                                activation=LeakyReLU(leaky_relu_alpha), use_bias=True)(
            focus_flux_branch)
        focus_flux_branch = keras.layers.MaxPooling1D(pool_size=5, strides=3)(focus_flux_branch)
        focus_flux_branch = keras.layers.Dropout(rate=0.1)(focus_flux_branch)
        focus_flux_branch = keras.layers.Conv1D(filters=32, kernel_size=3, padding="same",
                                                activation=LeakyReLU(leaky_relu_alpha), use_bias=True)(
            focus_flux_branch)
        focus_flux_branch = keras.layers.MaxPooling1D(pool_size=3, strides=3)(focus_flux_branch)
        focus_flux_branch = keras.layers.Dense(16, activation=LeakyReLU(leaky_relu_alpha))(focus_flux_branch)
        focus_flux_branch = keras.layers.Dropout(rate=0.1)(focus_flux_branch)
        focus_flux_branch = keras.layers.LayerNormalization()(focus_flux_branch)
        return focus_flux_input, focus_flux_branch

    def get_focus_flux_model_branch(self):
        leaky_relu_alpha = 0.01
        odd_flux_input, odd_flux_branch = self.get_focus_flux_branch("focus_odd_flux_branch")
        even_flux_input, even_flux_branch = self.get_focus_flux_branch("focus_even_flux_branch")
        harmonic_odd_flux_input, harmonic_odd_flux_branch = self.get_focus_flux_branch("focus_harmonic_odd_flux_branch")
        harmonic_even_flux_input, harmonic_even_flux_branch = self.get_focus_flux_branch("focus_harmonic_even_flux_branch")
        subharmonic_odd_flux_input, subharmonic_odd_flux_branch = self.get_focus_flux_branch(
            "focus_subharmonic_odd_flux_branch")
        subharmonic_even_flux_input, subharmonic_even_flux_branch = self.get_focus_flux_branch(
            "focus_subharmonic_even_flux_branch")
        odd_flux_branch = keras.layers.Add()(
            [odd_flux_branch, harmonic_odd_flux_branch, subharmonic_odd_flux_branch])
        even_flux_branch = keras.layers.Add()(
            [even_flux_branch, harmonic_even_flux_branch, subharmonic_even_flux_branch])
        odd_flux_branch = keras.layers.Dense(16, activation=LeakyReLU(leaky_relu_alpha))(odd_flux_branch)
        odd_flux_branch = keras.layers.Dropout(rate=0.1)(odd_flux_branch)
        even_flux_branch = keras.layers.Dense(16, activation=LeakyReLU(leaky_relu_alpha))(even_flux_branch)
        even_flux_branch = keras.layers.Dropout(rate=0.1)(even_flux_branch)
        odd_flux_branch = keras.layers.Dense(32, activation='linear')(odd_flux_branch)
        even_flux_branch = keras.layers.Dense(32, activation='linear')(even_flux_branch)
        odd_flux_branch = keras.layers.LayerNormalization()(odd_flux_branch)
        even_flux_branch = keras.layers.LayerNormalization()(even_flux_branch)
        focus_flux_branch = keras.layers.Add()([odd_flux_branch, even_flux_branch])
        focus_flux_branch = keras.layers.Dense(24, activation=LeakyReLU(leaky_relu_alpha))(focus_flux_branch)
        focus_flux_branch = keras.layers.Dense(16, activation='linear')(focus_flux_branch)
        input = [odd_flux_input, even_flux_input, harmonic_odd_flux_input, harmonic_even_flux_input,
                 subharmonic_odd_flux_input, subharmonic_even_flux_input]
        focus_flux_branch = keras.Model(inputs=input, outputs=focus_flux_branch)
        return focus_flux_branch

    def get_singletransit_tpf_model(self):
        video_image_width = 13
        video_image_height = 13
        video_image_channels = 1
        sequences_per_video = 100
        tpf_model_input = keras.Input(
            shape=(video_image_height, video_image_width, sequences_per_video, video_image_channels),
            name="tpf_input")
        tpf_model = keras.layers.SpatialDropout3D(rate=0.3)(tpf_model_input)
        tpf_model = keras.layers.Conv3D(filters=100, kernel_size=(3, 3, 5), strides=(1, 1, 3), activation="relu")(
            tpf_model)
        tpf_model = keras.layers.SpatialDropout3D(rate=0.2)(tpf_model)
        tpf_model = keras.layers.Conv3D(filters=200, kernel_size=(3, 3, 5), strides=(1, 1, 3), activation="relu")(
            tpf_model)
        tpf_model = keras.layers.SpatialDropout3D(rate=0.1)(tpf_model)
        tpf_model = keras.layers.MaxPooling3D(pool_size=(5, 5, 10), strides=(3, 3, 6), padding='same')(tpf_model)
        tpf_model = keras.layers.Dense(200, activation="relu")(tpf_model)
        tpf_model = keras.layers.Dense(100, activation="relu")(tpf_model)
        tpf_model = keras.layers.Dense(20, activation="relu")(tpf_model)
        tpf_model = keras.layers.Flatten()(tpf_model)
        return keras.Model(inputs=tpf_model_input, outputs=tpf_model)

    def get_singletransit_motion_centroids_model(self):
        mc_input = keras.Input(
            shape=(100, 4),
            name="motion_centroids_input")
        mc_model = keras.layers.Conv1D(filters=50, kernel_size=3, strides=3, activation="relu", use_bias=True,
                                       padding='same')(mc_input)
        mc_model = keras.layers.SpatialDropout1D(rate=0.3)(mc_model)
        mc_model = keras.layers.Conv1D(filters=100, kernel_size=5, strides=5, activation="relu", use_bias=True,
                                       padding='same')(mc_model)
        mc_model = keras.layers.SpatialDropout1D(rate=0.2)(mc_model)
        mc_model = keras.layers.MaxPooling1D(pool_size=5, strides=3, padding='same')(mc_model)
        mc_model = keras.layers.Dense(50, activation="relu")(mc_model)
        mc_model = keras.layers.Dense(20, activation="relu")(mc_model)
        mc_model = keras.layers.Flatten()(mc_model)
        return keras.Model(inputs=mc_input, outputs=mc_model)

    def get_singletransit_bckflux_model(self):
        bck_input = keras.Input(shape=(100, 1), name="bck_input")
        bck_model = keras.layers.Conv1D(filters=25, kernel_size=2, strides=2, activation="relu", use_bias=True,
                                        padding='same')(bck_input)
        bck_model = keras.layers.SpatialDropout1D(rate=0.3)(bck_model)
        bck_model = keras.layers.Conv1D(filters=50, kernel_size=3, strides=3, activation="relu", use_bias=True,
                                        padding='same')(bck_model)
        bck_model = keras.layers.SpatialDropout1D(rate=0.2)(bck_model)
        bck_model = keras.layers.MaxPooling1D(pool_size=5, strides=3, padding='same')(bck_model)
        bck_model = keras.layers.Dense(50, activation="relu")(bck_model)
        bck_model = keras.layers.Dense(10, activation="relu")(bck_model)
        bck_model = keras.layers.Flatten()(bck_model)
        return keras.Model(inputs=bck_input, outputs=bck_model)

    def get_singletransit_flux_model(self):
        bck_input = keras.Input(shape=(100, 1), name="flux_input")
        bck_model = keras.layers.Conv1D(filters=25, kernel_size=2, strides=2, activation="relu", use_bias=True,
                                        padding='same')(bck_input)
        bck_model = keras.layers.SpatialDropout1D(rate=0.3)(bck_model)
        bck_model = keras.layers.Conv1D(filters=50, kernel_size=3, strides=3, activation="relu", use_bias=True,
                                        padding='same')(bck_model)
        bck_model = keras.layers.SpatialDropout1D(rate=0.2)(bck_model)
        bck_model = keras.layers.MaxPooling1D(pool_size=5, strides=3, padding='same')(bck_model)
        bck_model = keras.layers.Dense(50, activation="relu")(bck_model)
        bck_model = keras.layers.Dense(10, activation="relu")(bck_model)
        bck_model = keras.layers.Flatten()(bck_model)
        return keras.Model(inputs=bck_input, outputs=bck_model)

    def get_single_transit_model(self):
        tpf_branch = self.get_singletransit_tpf_model()
        mc_branch = self.get_singletransit_motion_centroids_model()
        bck_branch = self.get_singletransit_bckflux_model()
        flux_branch = self.get_singletransit_flux_model()
        final_branch = keras.layers.concatenate(
            [tpf_branch.output, mc_branch.output, bck_branch.output, flux_branch.output], axis=1)
        final_branch = keras.layers.Dense(64, activation="relu", name="final-dense1")(final_branch)
        final_branch = keras.layers.Dense(32, activation="relu", name="final-dense2")(final_branch)
        final_branch = keras.layers.Dense(1, activation="softmax", name="final-dense-softmax")(final_branch)
        inputs = tpf_branch.inputs + mc_branch.inputs + bck_branch.inputs + flux_branch.inputs
        model = keras.Model(inputs=inputs, outputs=final_branch, name="mnist_model")
        keras.utils.vis_utils.plot_model(model, "IATSON_model.png", show_shapes=True)

    def get_model(self):
        leaky_relu_alpha = 0.01
        stellar_model_input = keras.Input(shape=(11, 1), name="stellar_model")
        stellar_model_branch = keras.layers.Dense(16, activation=LeakyReLU(leaky_relu_alpha), name="stellar-first")(stellar_model_input)
        stellar_model_branch = keras.layers.Dropout(rate=0.1, name="stellar-first-dropout-0.1")(stellar_model_branch)
        stellar_model_branch = keras.layers.Dense(16, activation="relu", name="stellar-refinement")(
            stellar_model_branch)
        flux_model_branch = self.get_flux_model_branch()
        focus_flux_model_branch = self.get_focus_flux_model_branch()
        final_branch = keras.layers.Concatenate(axis=1)(
            [stellar_model_branch, flux_model_branch.output, focus_flux_model_branch.output])
        final_branch = keras.layers.Dense(64, activation=LeakyReLU(leaky_relu_alpha), name="final-dense1")(final_branch)
        final_branch = keras.layers.Dense(16, activation=LeakyReLU(leaky_relu_alpha), name="final-dense2")(final_branch)
        final_branch = keras.layers.Dense(3, activation="softmax", name="final-dense-softmax")(final_branch)
        inputs = [stellar_model_input] + flux_model_branch.inputs + focus_flux_model_branch.inputs
        model = keras.Model(inputs=inputs, outputs=final_branch, name="mnist_model")
        keras.utils.vis_utils.plot_model(model, "IATSON.png", show_shapes=True)

IATSON().get_model()