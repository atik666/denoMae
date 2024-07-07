import tensorflow as tf
import random
import os
import pandas as pd


def limit_data(config, n=100000):
    a = []

    rgb_dir = config['dataset_path']
    depth_dir = config['depth_path']
    normal_dir = config['thermal_path']

    print(f"RGB Directory: {rgb_dir}")
    print(f"Depth Directory: {depth_dir}")
    print(f"Normal Directory: {normal_dir}")

    files_list = os.listdir(rgb_dir)
    random.shuffle(files_list)

    for folder_name in files_list:
        rgb_folder = os.path.join(rgb_dir, folder_name)
        depth_folder = os.path.join(depth_dir, folder_name)
        normal_folder = os.path.join(normal_dir, folder_name)

        if not os.path.isdir(rgb_folder) or not os.path.isdir(depth_folder) or not os.path.isdir(normal_folder):
            print(f"Missing folder for class '{folder_name}'. Check RGB, Depth, and Normal directories.")
            continue

        rgb_files = os.listdir(rgb_folder)
        for k, rgb_file in enumerate(rgb_files):
            if k >= n:
                break

            base_name = rgb_file.split('_rgb_')[0]
            depth_file_name = f"{base_name}_depth_{rgb_file.split('_rgb_')[1]}"
            normal_file_name = f"{base_name}_normal_{rgb_file.split('_rgb_')[1]}"

            rgb_path = os.path.join(rgb_folder, rgb_file)
            depth_path = os.path.join(depth_folder, depth_file_name)
            normal_path = os.path.join(normal_folder, normal_file_name)

            if os.path.exists(depth_path) and os.path.exists(normal_path):
                a.append((rgb_path, depth_path, normal_path, folder_name))
            else:
                if not os.path.exists(depth_path):
                    print(f"Depth file not found: {depth_path}")
                if not os.path.exists(normal_path):
                    print(f"Normal file not found: {normal_path}")

    df = pd.DataFrame(a, columns=['rgb', 'depth', 'normal', 'class'])
    print(f"Total image triplets found: {len(df)}")
    return df

class MultiModalDataGenerator:
    def __init__(self, df, config, subset):
        self.df = df
        self.batch_size = config['batch_size']
        self.target_size = (config['img_height'], config['img_width'])
        self.subset = subset
        self.config = config

        # Mapping class labels to integers
        self.class_names = sorted(self.df['class'].unique())
        self.class_to_index = {name: index for index, name in enumerate(self.class_names)}

        validation_views = ['KSFO Runway 19L', 'KLAX Runway 24R 19deg', 'KACY Runway 31 19deg', 'CYQB Runway 29 252deg', '6N7 Sealane 01 146deg', 'KLGB Runway 08L 146deg']
    
        if subset == 'training':
            df_train = self.df[~self.df['rgb'].str.contains('|'.join(validation_views))]
            self.df = df_train
            # self.df = self.df.sample(frac=1-config.val_split, random_state=config.seed)
        elif subset == 'validation':
            df_validation = self.df[self.df['rgb'].str.contains('|'.join(validation_views))]
            self.df = df_validation

        self.log_image_counts()

    def log_image_counts(self):
        self.num_images = len(self.df)
        counts_per_set = self.df['class'].value_counts()
        print(f"Total image sets in {self.subset} set:")
        print(counts_per_set)

    def preprocess_image(self, img_path, resize_shape):
        img = tf.io.read_file(img_path)
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.resize(img, resize_shape)
        img = img / 255.0
        return img

    def load_and_preprocess_images(self, row):
        rgb_img = self.preprocess_image(row['rgb'], self.target_size)
        depth_img = self.preprocess_image(row['depth'], self.target_size)
        normal_img = self.preprocess_image(row['normal'], self.target_size)
        # normal_img = self.preprocess_image(row['normal'], self.target_size) if self.config.normal else None
        return rgb_img, depth_img, normal_img

    def generate_data(self):
        def gen():
            for _, row in self.df.iterrows():
                rgb_img, depth_img, normal_img = self.load_and_preprocess_images(row)
                class_label = self.class_to_index[row['class']]
                one_hot_label = tf.one_hot(class_label, depth=self.config['num_classes'])

                yield (rgb_img, depth_img, normal_img), one_hot_label

        return tf.data.Dataset.from_generator(
            gen,
            output_signature=(
                (
                    tf.TensorSpec(shape=(self.target_size[0], self.target_size[1], 3), dtype=tf.float32),
                    tf.TensorSpec(shape=(self.target_size[0], self.target_size[1], 3), dtype=tf.float32),
                    tf.TensorSpec(shape=(self.target_size[0], self.target_size[1], 3), dtype=tf.float32)
                ),
            tf.TensorSpec(shape=(self.config['num_classes']), dtype=tf.float32)
            )
        ).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

def load_dataset(config):
    full_df = limit_data(config, config['num_img_lim'])

    train_generator = MultiModalDataGenerator(full_df, config, subset='training')
    validation_generator = MultiModalDataGenerator(full_df, config, subset='validation')

    return train_generator, validation_generator

