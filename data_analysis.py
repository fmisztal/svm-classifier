from ucimlrepo import fetch_ucirepo
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def visualize_quality_data():
    wine_quality = fetch_ucirepo(id=186)

    X = pd.DataFrame(wine_quality.data.features)
    y = pd.DataFrame(wine_quality.data.targets)
    y['quality'] = y['quality'].apply(lambda x: -1 if x <= 5 else 1)
    data = pd.concat([X, y], axis=1)

    column_names = data.columns.tolist()
    column_names.remove('quality')

    fig, axs = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle('Dataset with quality split in half', fontsize=16)
    for i, ax in enumerate(fig.get_axes()):
        if i < len(column_names):
            feature = column_names[i]
            sns.histplot(data=data, x=feature, hue='quality', kde=True, element='step', stat='density', common_norm=False,
                         ax=ax, palette='viridis')
            ax.set_title(feature)
            ax.set_ylabel('Density')
            ax.set_xlabel('Feature value')
        else:
            fig.delaxes(ax)
    plt.tight_layout()
    plt.show()


def visualize_quality_mean_data():
    wine_quality = fetch_ucirepo(id=186)

    X = pd.DataFrame(wine_quality.data.features)
    y = pd.DataFrame(wine_quality.data.targets)

    mean = round(sum(y['quality']) / len(y['quality']))
    y['quality'] = y['quality'].apply(lambda x: -1 if x <= mean else 1)
    data = pd.concat([X, y], axis=1)

    column_names = data.columns.tolist()
    column_names.remove('quality')

    fig, axs = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle('Dataset with quality divided by average', fontsize=16)
    for i, ax in enumerate(fig.get_axes()):
        if i < len(column_names):
            feature = column_names[i]
            sns.histplot(data=data, x=feature, hue='quality', kde=True, element='step', stat='density', common_norm=False,
                         ax=ax, palette='viridis')
            ax.set_title(feature)
            ax.set_ylabel('Density')
            ax.set_xlabel('Feature value')
        else:
            fig.delaxes(ax)
    plt.tight_layout()
    plt.show()


def visualize_color_data():
    df = pd.read_csv('wine_color.csv').drop(columns=['quality'])
    column_names = df.columns.tolist()
    column_names.remove('color')

    fig, axs = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle('Dataset with color', fontsize=16)
    for i, ax in enumerate(fig.get_axes()):
        if i < len(column_names):
            feature = column_names[i]
            sns.histplot(data=df, x=feature, hue='color', kde=True, element='step', stat='density', common_norm=False,
                         ax=ax, palette='viridis')
            ax.set_title(feature)
            ax.set_ylabel('Density')
            ax.set_xlabel('Feature value')
        else:
            fig.delaxes(ax)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    visualize_quality_data()
    visualize_quality_mean_data()
    visualize_color_data()