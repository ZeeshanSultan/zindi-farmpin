from invoke import task


@task
def create_baseline_dataset(ctx):
    from src.data.make_baseline_dataset import run
    print('Creating training set\n')
    run('train')
    print('\nCreating testing set\n')
    run('test')
    print('done!')

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


@task(help={
    'search': 'search term'
})
def competition_list(ctx, search=None):
    """
    List Kaggle competitions
    """
    cmd = ['kaggle competitions list']
    if search:
        cmd = cmd + ['-s {}'.format(search)]
    ctx.run(' '.join(cmd))


@task()
def competition_download_files(ctx):
    """
    Download Kaggle competition files to ./data/raw folder
    """

    cmd = 'wget -i urls.txt -P ./data/raw/'
    ctx.run(cmd)


@task(help={
    'path': 'path to file to submit',
    'message': 'message describing the submission',
    'competition': 'competition url prefix'
})
def competition_submit_files(ctx, path, message, competition):
    """
    Submit Kaggle competition files
    """
    cmd = 'kaggle competitions submit', '-c {} -f {} -m {}'.format(competition, path, message)
    ctx.run(cmd)