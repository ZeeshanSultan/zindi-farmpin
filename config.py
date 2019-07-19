import os

root_dir = os.path.dirname(os.path.abspath(__file__))

# Create data dir paths
data_dir = os.path.join(root_dir, 'data')
interim_data_dir = os.path.join(data_dir, 'interim')
processed_data_dir = os.path.join(data_dir, 'processed')
raw_data_dir = os.path.join(data_dir, 'raw')
subs_dir = os.path.join(data_dir, 'submissions')

if __name__ == '__main__':
    pass
