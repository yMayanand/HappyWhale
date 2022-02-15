import concurrent.futures

# write your function that will act as single unit of execution 
def func(path, df, idx):
    df.loc[idx, 'width'], df.loc[idx, 'height'] = Image.open(path).size


# through this block of code one can benefit from multi-threading
# this should be used mostly when func executes IO-bound tasks
with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = []
    for idx, path in enumerate(df['image_path']):
        futures.append(executor.submit(func, path, temp_df, idx))
    count = 0
    for future in concurrent.futures.as_completed(futures):
        print(f'\r{count}', end='')
        count += 1