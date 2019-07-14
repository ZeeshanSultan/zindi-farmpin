from invoke import task


@task(help={
    'ip': 'IP to listen on, defaults to *',
    'extra': 'Port to listen on, defaults to 8888',
})
def lab(ctx, ip='*', port=8888):
    """
    Launch Jupyter lab
    """
    cmd = ['jupyter lab', '--ip={}'.format(ip), '--port={}'.format(port)]
    ctx.run(' '.join(cmd))


@task(help={
    'ip': 'IP to listen on, defaults to *',
    'extra': 'Port to listen on, defaults to 8888',
})
def notebook(ctx, ip='*', port=8888):
    """
    Launch Jupyter notebook
    """
    cmd = ['jupyter notebook', '--ip={}'.format(ip), '--port={}'.format(port)]
    ctx.run(' '.join(cmd))


@task
def download_satellite_data(ctx):
    """
    Download Kaggle competition files to ./data/raw folder
    """

    # Download zips
    cmd = 'wget -i urls.txt -P ./data/raw/'
    ctx.run(cmd)

    # Download extra zips
    cmd = 'wget -i urls_JFP.txt -P ./data/raw/'
    ctx.run(cmd)

    # Unzip and remove zips
    cmd = "cd ./data/raw && unzip '*.zip' && rm *.zip"
    ctx.run(cmd)


@task
def reorder_dataset(ctx):
    from src.data.restructure_data import run
    print('Re-ordering dataset --> output in data/interim/images folder \n')
    run()

@task
def create_stacked_masks_dataset(ctx):
    from src.data.make_stacked_masks_dataset import run
    print('Creating stacked masks dataset. Output saved under data/processed/ \n')
    run()

@task
def create_masks_dataset(ctx):
    from src.data.make_masks_dataset import run
    print('Creating stacked masks dataset. Output saved under data/interim/masks/ \n')

    print('Creating training set\n')
    run('train')

    print('Creating testing set\n')
    run('test')

@task
def create_baseline_dataset(ctx):
    from src.data.make_features_dataset import run
    print('Creating training set\n')
    run('train')
    print('\nCreating testing set\n')
    run('test')
    print('done!')
