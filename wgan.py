import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import random


tf.random.set_seed(16)
np.random.seed(16)
random.seed(16)

data=tf.keras.datasets.mnist.load_data()   #download MNIST dataset
(train_images,train_labels),(test_images,test_labels)=data    #convert all the data into 4 tuples

train_images=np.expand_dims(train_images,axis=-1)
train_images=train_images.astype('float32')
train_images=((train_images/255.0)*2)-1  #normalize the train images to the range [-1, 1]

test_images=np.expand_dims(test_images,axis=-1)
test_images=test_images.astype('float32')
test_images=((test_images/255.0)*2)-1  #normalize the test images to the range [-1, 1]



G=tf.keras.Sequential([
    tf.keras.layers.Dense(units=12544,input_shape=(100,)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.Reshape((7,7,256)),
    tf.keras.layers.Conv2DTranspose(filters=128,kernel_size=(3,3),padding="same",strides=(2,2)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.Conv2DTranspose(filters=1,kernel_size=(3,3),activation="tanh",padding="same",strides=(2,2))
])

G.summary()

D=tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3),padding="same",strides=(2,2),input_shape=(28,28,1)),
    tf.keras.layers.LayerNormalization(),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.Conv2D(filters=128,kernel_size=(3,3),padding="same",strides=(2,2)),
    tf.keras.layers.LayerNormalization(),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=256,activation="relu"),  
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(units=1)
])

D.summary()

from scipy.linalg import sqrtm

GP_WEIGHT = 10.0
bestfid = float('inf')
save_path = 'best_generator.weights.h5'
epochs = 100
latent_dim = 100
D_loss_mean = []
G_loss_mean = []
D_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5)
G_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5)

n_critic = 1

@tf.function
def Train_D_Step(real_images):
    # Determina la dimensione del batch dinamicamente
    current_batch_size = tf.shape(real_images)[0]

    # Genera rumore per le immagini fake
    noise = tf.random.normal(shape=(current_batch_size, latent_dim))

    with tf.GradientTape() as disc_tape:
        fake_images = G(noise, training=True)
        real_output = D(real_images, training=True)
        fake_output = D(fake_images, training=True)

        # Loss 
        disc_wgan_loss = tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)

        # Gradient Penalty
        alpha = tf.random.uniform(shape=[current_batch_size, 1, 1, 1], minval=0., maxval=1.)
        interpolated_images = alpha * real_images + (1 - alpha) * fake_images

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated_images)
            pred = D(interpolated_images, training=True)

        grads = gp_tape.gradient(pred, [interpolated_images])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gradient_penalty = tf.reduce_mean((norm - 1.0)**2)

        disc_loss = disc_wgan_loss + GP_WEIGHT * gradient_penalty

    D_Gradient = disc_tape.gradient(disc_loss, D.trainable_variables)
    D_optimizer.apply_gradients(zip(D_Gradient, D.trainable_variables))

    return disc_loss

@tf.function
def Train_G_Step(current_batch_size):
    with tf.GradientTape() as gen_tape:
        noise = tf.random.normal(shape=(current_batch_size, latent_dim))
        fake_images = G(noise, training=True)
        fake_output = D(fake_images, training=True)

        gen_loss = -tf.reduce_mean(fake_output)

    G_Gradient = gen_tape.gradient(gen_loss, G.trainable_variables)
    G_optimizer.apply_gradients(zip(G_Gradient, G.trainable_variables))

    return gen_loss

batch_size = 128
dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(len(train_images)).batch(batch_size)

step_counter = 0

for epoch in range(epochs):
    epoch_d_loss = []
    epoch_g_loss = []

    g_loss = 0.0

    for image_batch in dataset:
        step_counter += 1

        # train D
        d_loss = Train_D_Step(image_batch)
        epoch_d_loss.append(d_loss.numpy())

        # train G
        if step_counter % n_critic == 0:
            current_batch_sz = tf.shape(image_batch)[0]
            g_loss = Train_G_Step(current_batch_sz)

        if tf.is_tensor(g_loss):
            epoch_g_loss.append(g_loss.numpy())
        else:
            epoch_g_loss.append(g_loss)

    avg_d = np.mean(epoch_d_loss)
    avg_g = np.mean(epoch_g_loss)

    D_loss_mean.append(avg_d)
    G_loss_mean.append(avg_g)

    print(f"Epoch {epoch + 1}/{epochs} - D Loss: {avg_d:.4f}, G Loss: {avg_g:.4f}")

print("\nTraining done.")


import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FixedLocator

epochs = 100

selected_epochs = [1] + list(range(5, epochs + 1, 5))

selected_D = [D_loss_mean[i-1] for i in selected_epochs]
selected_G = [G_loss_mean[i-1] for i in selected_epochs]

plt.figure(figsize=(10, 6))
ax = plt.gca()

ax.plot(selected_epochs, selected_G, color="#1F77B4", marker="s", markersize=3.5,
        lw=1.6, clip_on=False, label="Generator Loss")

ax.plot(selected_epochs, selected_D, color="#EB3300", marker="o", markersize=3.5,
        lw=1.6, clip_on=False, label="Critic Loss")


ax.set_yscale('symlog', linthresh=1)

# Griglia e Spines (Bordi)
ax.yaxis.grid(True, linestyle="--", linewidth=0.5, color="#B1B3B3")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_color("#B1B3B3")

ax.tick_params(axis="x", color="#B1B3B3", labelcolor="#3F4443", labelsize=8)
ax.tick_params(axis="y", color="#B1B3B3", labelcolor="#3F4443")

ax.set_xticks(selected_epochs)

ax.legend(loc="center right", frameon=False, fontsize=12, prop={"family": "serif"})

fig = plt.gcf()
fig.text(0.5, 0.95, "WGAN Losses", ha="center", va="top", fontsize=20, color="#3F4443")


plt.show()

latent_dim = 100

def generate_and_display_images(model, num_images_to_generate):
    noise = tf.random.normal([num_images_to_generate, latent_dim])
    generated_images = model(noise, training=False)
    generated_images = (generated_images + 1) / 2.0

    grid_size = int(np.sqrt(num_images_to_generate))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(8, 8))

    axes = axes.flatten()

    for i in range(num_images_to_generate):
        img = generated_images[i, :, :, 0]

        axes[i].imshow(img, cmap='gray')
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

generate_and_display_images(G, 64)

import numpy as np
import tensorflow as tf
from scipy.linalg import sqrtm


NUM_IMAGES_FID = 10000
BATCH_SIZE = 16
LATENT_DIM = 100

def calculate_fid_score(mu1, sigma1, mu2, sigma2):
    #Calcola il FID Score date le statistiche (media e covarianza).
    ssdiff = np.sum((mu1 - mu2)**2.0)

    covmean = sqrtm(sigma1.dot(sigma2))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

def preprocess_image_mapping(image):

    image = (image * 127.5) + 127.5

    image = tf.image.resize(image, [299, 299], method='bilinear')

    image = tf.image.grayscale_to_rgb(image)

    image = tf.keras.applications.inception_v3.preprocess_input(image)

    return image

def get_inception_statistics(model, images_tensor, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices(images_tensor)
    dataset = dataset.map(preprocess_image_mapping, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    act = model.predict(dataset, verbose=1)

    # Calcolo statistiche
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)

    return mu, sigma


#  Caricamento InceptionV3
print("Loading images")
inception_model = tf.keras.applications.InceptionV3(
    include_top=False,
    pooling='avg',
    input_shape=(299, 299, 3)
)

# Calcolo statistiche immagini REALI
real_subset = train_images[:NUM_IMAGES_FID]
real_subset = tf.cast(real_subset, tf.float32)
if len(real_subset.shape) == 3:
    real_subset = tf.expand_dims(real_subset, axis=-1)

mu_real, sigma_real = get_inception_statistics(inception_model, real_subset, BATCH_SIZE)

# Calcolo statistiche immagini FAKE

fake_images_list = []
num_batches = int(np.ceil(NUM_IMAGES_FID / BATCH_SIZE))

for i in range(num_batches):
    current_batch_size = min(BATCH_SIZE, NUM_IMAGES_FID - len(fake_images_list) * BATCH_SIZE)
    if current_batch_size <= 0: break

    noise = tf.random.normal([current_batch_size, LATENT_DIM])
    generated_batch = G(noise, training=False)
    fake_images_list.append(generated_batch)

fake_images_tensor = tf.concat(fake_images_list, axis=0)

mu_fake, sigma_fake = get_inception_statistics(inception_model, fake_images_tensor, BATCH_SIZE)

# Calcolo finale
fid = calculate_fid_score(mu_real, sigma_real, mu_fake, sigma_fake)

print(f"FID SCORE: {fid:.4f}")


import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import random
import pandas as pd

tf.random.set_seed(16)
np.random.seed(16)
random.seed(16)
try:
    dataframe = yf.download('^GSPC', start='2000-01-01', end='2023-12-31', progress=False)
    print("download done")
except Exception as e:
    print(f"error during download: {e}")
    exit()

log_returns=np.log(dataframe['Close']/dataframe['Close'].shift(1)).dropna()

data = log_returns.values
print(f"\nloaded {len(data)} data.")

sequence_lenth=60
step_size=1
dataset_list=[]

for i in range(0,len(data)-sequence_lenth+step_size):
    x=[]
    for j in range(sequence_lenth):
        x.append(data[j+i])
    dataset_list.append(x)

dataset = np.array(dataset_list)
dataset = np.reshape(dataset, (-1, 60, 1))

min_val = dataset.min()
max_val = dataset.max()
print("min =" + str(min_val)+" max = " + str(max_val))

dataset=2*(dataset-min_val)/(max_val-min_val)-1  #data normalization with center in zero
train_sequence = dataset.astype('float32')


G=tf.keras.Sequential([
    tf.keras.layers.Dense(units=1500,input_shape=(100,)),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.Reshape((15,100)),
    tf.keras.layers.Conv1DTranspose(filters=256,kernel_size=3,padding="same",strides=2),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.Conv1DTranspose(filters=1,kernel_size=3,activation="tanh",padding="same",strides=2),
])

G.summary()

D=tf.keras.Sequential([
    tf.keras.layers.Conv1D(filters=128,kernel_size=3,padding="same",strides=2,input_shape=(60,1)),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.Conv1D(filters=256,kernel_size=3,padding="same",strides=2),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=256,activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(units=1)
])

D.summary()

from scipy.linalg import sqrtm

GP_WEIGHT=10.0 # Standard value for the Gradient Penalty weight
batch_size=256
epochs=100
latent_dim=100
D_loss_mean=[]
G_loss_mean=[]
D_optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001, beta_1=0.5,clipvalue=0.01)
G_optimizer=tf.keras.optimizers.Adam(learning_rate=0.000001, beta_1=0.5,clipvalue=0.01)
dataset_tf=tf.data.Dataset.from_tensor_slices(train_sequence).shuffle(len(train_sequence)).batch(batch_size,drop_remainder=True)

@tf.function
def train_critic_step(real_sequence):
    batch_size = tf.shape(real_sequence)[0]
    noise = tf.random.normal(shape=(batch_size, latent_dim))

    with tf.GradientTape() as disc_tape:
        fake_sequence = G(noise, training=True)
        real_output = D(real_sequence, training=True)
        fake_output = D(fake_sequence, training=True)

        # Loss WGAN e Gradient Penalty 
        disc_wgan_loss = tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)
        alpha = tf.random.uniform(shape=[batch_size, 1, 1], minval=0., maxval=1.)
        interpolated_sequence = alpha * real_sequence + (1 - alpha) * fake_sequence
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated_sequence)
            pred = D(interpolated_sequence, training=True)
        grads = gp_tape.gradient(pred, [interpolated_sequence])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2]))
        gradient_penalty = tf.reduce_mean((norm - 1.0) ** 2)
        disc_loss = disc_wgan_loss + GP_WEIGHT * gradient_penalty

    D_Gradient = disc_tape.gradient(disc_loss, D.trainable_variables)
    D_optimizer.apply_gradients(zip(D_Gradient, D.trainable_variables))
    return disc_loss

@tf.function
def train_generator_step(batch_size):
    noise = tf.random.normal(shape=(batch_size, latent_dim))
    with tf.GradientTape() as gen_tape:
        fake_sequence = G(noise, training=True)
        fake_output = D(fake_sequence, training=True)
        gen_loss = -tf.reduce_mean(fake_output)

    G_Gradient = gen_tape.gradient(gen_loss, G.trainable_variables)
    G_optimizer.apply_gradients(zip(G_Gradient, G.trainable_variables))
    return gen_loss

n_critic = 5

for epoch in range(epochs):
    epoch_d_loss = []
    epoch_g_loss = []

   
    for step, sequence_batch in enumerate(dataset_tf):

        # Il Critico viene aggiornato ad ogni step
        d_loss = train_critic_step(sequence_batch)
        epoch_d_loss.append(d_loss.numpy())

        # Il Generatore viene aggiornato solo ogni n_critic steps
        if step % n_critic == 0:
            g_loss = train_generator_step(batch_size)
            epoch_g_loss.append(g_loss.numpy())

    # Calcolo delle medie per le statistiche dell'epoca
    avg_d = np.mean(epoch_d_loss)
    avg_g = np.mean(epoch_g_loss) if len(epoch_g_loss) > 0 else 0

    D_loss_mean.append(avg_d)
    G_loss_mean.append(avg_g)


    avg_d=np.mean(epoch_d_loss)
    avg_g=np.mean(epoch_g_loss)

    D_loss_mean.append(avg_d)
    G_loss_mean.append(avg_g)

    print(f"Epoch {epoch + 1}/{epochs} - D Loss: {avg_d:.4f}, G Loss: {avg_g:.4f}")


#plot of G and L mean loss for all epochs
from matplotlib.ticker import MultipleLocator, FixedLocator

plt.figure(figsize=(10,6))
xvalues=[1,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]
D_plot=[]
G_plot=[]
for el in xvalues:
    D_plot.append(D_loss_mean[el-1])
    G_plot.append(G_loss_mean[el-1])

ax=plt.gca()
ax.set_ylim(-0.8, 30)
ax.plot(xvalues,D_plot,color="#EB3300", marker="o",markersize=3.5,lw=1.6,clip_on=False)
ax.yaxis.grid(True,linestyle="--",linewidth=0.5,color="#B1B3B3")
ax.set_yscale('symlog', linthresh=1)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_color("#B1B3B3")
ax.tick_params(axis="x",color="#B1B3B3",labelcolor="#3F4443")
ax.tick_params(axis="y",color="#B1B3B3",labelcolor="#3F4443")
ax.xaxis.label.set_color("#B1B3B3")

ticks = [1] + list(range(5, 101, 5))

ax.set_xticks(ticks) 
ax.tick_params(axis='x', labelsize=8, colors="#3F4443")  

ax.plot(xvalues, G_plot, color="#1F77B4", marker="s", markersize=3.5,
        lw=1.6, clip_on=False, label="Generator Loss")

ax.plot(xvalues, D_plot, color="#EB3300", marker="o", markersize=3.5,
        lw=1.6, clip_on=False, label="Critic Loss")

ax.legend(loc="upper right",frameon=False,fontsize=12,prop={"family":"serif"})


fig=plt.gcf()
fig.text(0.5,0.95,"WGAN Losses",ha="center",va="top",fontsize=20,color="#3F4443")
plt.show()

# Generazione dato Finto
noise = tf.random.normal(shape=(1, latent_dim))
generated_sequence_normalized = G(noise, training=False).numpy().squeeze()
# De-normalizzazione dati finti
generated_sequence_real = ((generated_sequence_normalized + 1) * (max_val - min_val) / 2) + min_val

# Selezione dato Reale 
real_sample_index = np.random.randint(0, len(train_sequence))
real_sequence_normalized = train_sequence[real_sample_index].squeeze()
# De-normalizzazione dato reale
real_sequence_real = ((real_sequence_normalized + 1) * (max_val - min_val) / 2) + min_val

y_min = np.minimum(real_sequence_real.min(), generated_sequence_real.min()) * 1.1
y_max = np.maximum(real_sequence_real.max(), generated_sequence_real.max()) * 1.1

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

# Grafico Reale
ax1.plot(real_sequence_real, color='#2c3e50', linewidth=1.2)
ax1.fill_between(range(len(real_sequence_real)), real_sequence_real, color='#2c3e50', alpha=0.15)
ax1.axhline(0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
ax1.set_title('Real Sequence Sample', fontsize=14, fontweight='bold', color='#2c3e50')
ax1.set_xlabel('Days', fontsize=10, color='dimgray')
ax1.set_ylabel('Log Return', fontsize=10, color='dimgray')
ax1.set_ylim(y_min, y_max)

# Grafico Generato
ax2.plot(generated_sequence_real, color='#1f77b4', linewidth=1.2)
ax2.fill_between(range(len(generated_sequence_real)), generated_sequence_real, color='#1f77b4', alpha=0.15)
ax2.axhline(0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
ax2.set_title('WGAN Generated Sample', fontsize=14, fontweight='bold',color='#2c3e50')
ax2.set_xlabel('Days', fontsize=10, color='dimgray')

for ax in [ax1, ax2]:
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('lightgray')
    ax.spines['bottom'].set_color('lightgray')
    ax.tick_params(axis='both', colors='dimgray', labelsize=9)

plt.tight_layout()
plt.show()


from tqdm import tqdm
import pandas as pd


NUM_SAMPLES = 5000  # Numero di sequenze da generare

all_generated_sequences = []

print(f"Generazione di {NUM_SAMPLES} campioni dal Generatore...")
for _ in tqdm(range(NUM_SAMPLES)):

    noise = tf.random.normal(shape=(1, latent_dim))         # Crea rumore casuale
    # Genera una sequenza normalizzata
    generated_sequence_normalized = G(noise, training=False)

    all_generated_sequences.append(generated_sequence_normalized.numpy())       # Aggiunge la sequenza alla lista


# Concatena tutte le sequenze in un unico grande array NumPy
generated_dataset_normalized = np.concatenate(all_generated_sequences)

generated_dataset_real=((generated_dataset_normalized+1)*(max_val-min_val)/2)+min_val    # denormalizzazione

# Appiattisce l'array per ottenere una singola serie di rendimenti giornalieri

generated_returns_flat = generated_dataset_real.flatten()

print(f"Mean:  {np.mean(data):.6f}  |   {np.mean(generated_returns_flat):.6f}")
print(f"Standard Deviation:{np.std(data):.6f}  |   {np.std(generated_returns_flat):.6f}")
print(f"Excess Kurtosis: {pd.Series(data.flatten()).kurtosis():.6f}  |   {pd.Series(generated_returns_flat).kurtosis():.6f}")


#linear scale
fig, ax = plt.subplots(figsize=(12, 7))

ax.hist(data, bins=200, density=True, label='Real Data',
        color='#d62728', histtype='step', linewidth=1.5)

ax.hist(generated_returns_flat, bins=200, density=True, label='Generated Data',
        color="#041DFF", alpha=0.6)

fig.suptitle('Distribution of Log-Returns Linear Scale', fontsize=18,color="#454142")

ax.set_xlabel('Log Return', fontsize=12, color='dimgray')
ax.set_ylabel('Probability Density', fontsize=12, color='dimgray')
ax.grid(True, linestyle='--', color='lightgray', alpha=0.7)
ax.set_xlim(-0.15,0.15)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_color('lightgray')

ax.tick_params(axis='x', colors='dimgray')
ax.tick_params(axis='y', colors='dimgray')

legend = ax.legend(frameon=True)
legend.get_frame().set_facecolor('white')
legend.get_frame().set_alpha(0.8)

plt.show()

# log scale
fig, ax = plt.subplots(figsize=(12, 7))

ax.hist(data, bins=200, density=True, label='Real Data',
        color='#d62728', histtype='step', linewidth=1.5)

ax.hist(generated_returns_flat, bins=200, density=True, label='Generated Data',
        color="#041DFF", alpha=0.6)

fig.suptitle('Distribution of Log-Returns Log Scale', fontsize=18,color="#454142")

ax.set_xlabel('Log Return', fontsize=12, color='dimgray')
ax.set_ylabel('Probability Density', fontsize=12, color='dimgray')
ax.grid(True, linestyle='--', color='lightgray', alpha=0.7)
ax.set_yscale('log')
ax.set_xlim(-0.15,0.15)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_color('lightgray')

ax.tick_params(axis='x', colors='dimgray')
ax.tick_params(axis='y', colors='dimgray')

legend = ax.legend(frameon=True)
legend.get_frame().set_facecolor('white')
legend.get_frame().set_alpha(0.8)
