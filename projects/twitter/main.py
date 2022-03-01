import twint

if __name__ == "__main__":
    c = twint.Config()
    username = "elonmuskbooks"
    c.Username = username
    c.Limit = 20
    c.Pandas = True

    twint.run.Followers(c)

    df = twint.storage.panda.Follow_df
    print(df.head())
