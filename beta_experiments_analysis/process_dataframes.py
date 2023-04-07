import matplotlib.pyplot as plt
import seaborn as sns

from parse_beta_experiments import parse_files


def main():
    df, total_pixels_noised = parse_files()
    total_pixels_unnoised_approx = total_pixels_noised * 9
    df["percentage_detected"] = df["correctly_detected"] / total_pixels_noised
    df["percentage_false_alarm"] = df["false_alarm"] / total_pixels_unnoised_approx
    # Co me zaujima: body kde percentage_false_alarm je mensie ako 0.05
    # a percentage_detected je vacsie ako 0.95
    filtered_df = df.query("percentage_false_alarm < 0.05 and percentage_detected > 0.95")
    # print count of rows per beta
    print(filtered_df.groupby("beta").count())
    sns.boxplot(df, x="beta", y="percentage_detected")
    plt.show()


if __name__ == "__main__":
    main()
