import pandas as pd


if __name__ == "__main__":
    names = pd.read_csv(
        "ncvoter_Statewide.txt",
        sep="\t",
        encoding="unicode_escape",
    )
    # only keep non unique names
    counts = names["first_name"].value_counts().reset_index()
    non_unique_names = counts[counts["count"] > 1]["first_name"].tolist()
    names = names[names["first_name"].isin(non_unique_names)]

    # get average age per name
    age = (
        names.groupby(["first_name"])
        .agg(
            {
                "age_at_year_end": "mean",
            }
        )
        .reset_index()
    )

    # get names with strong binary gender, race and ethnicity associations
    binary_gender_prop = (
        names.groupby(["first_name"])[
            ["race_code", "ethnic_code", "gender_code"]
        ]
        .value_counts(normalize=True)
        .reset_index()
    )
    binary_gender = binary_gender_prop[binary_gender_prop["proportion"] > 0.9]
    binary_gender = binary_gender[
        binary_gender["gender_code"].isin(["F", "M"])
    ]

    # get names which are equally associated with male and female (and unknown) gender
    non_binary_gender_names = (
        names.groupby(["first_name"])
        .agg({"gender_code": pd.Series.mode})
        .reset_index()
    )
    non_binary_gender_names["gender_code"] = non_binary_gender_names[
        "gender_code"
    ].apply(lambda x: str(x))
    non_binary_gender_names = non_binary_gender_names[
        non_binary_gender_names["gender_code"].isin(
            ["['F' 'M' 'U']", "['F' 'M']"]
        )
    ]["first_name"].tolist()

    # get names with strong race and ethnicity associations
    race_eth_prop = (
        names.groupby(["first_name"])[["race_code", "ethnic_code"]]
        .value_counts(normalize=True)
        .reset_index()
    )

    race_eth = race_eth_prop[race_eth_prop["proportion"] > 0.9]
    # out of those names, get non-binary names
    non_binary_gender = race_eth[
        race_eth["first_name"].isin(non_binary_gender_names)
    ]
    non_binary_gender["gender_code"] = "N"

    # merge gendered and non-binary names
    processed_names = pd.concat(
        [binary_gender, non_binary_gender], ignore_index=True
    )
    processed_names = pd.merge(processed_names, age, on="first_name").drop(
        columns=["proportion"]
    )
    processed_names.to_csv("ncvoter_Statewide_processed.csv")
