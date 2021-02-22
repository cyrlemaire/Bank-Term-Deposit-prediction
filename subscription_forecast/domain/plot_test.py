import seaborn as sb
import matplotlib.pyplot as plt


def boxplot(data, col_a, col_b):
    sb.boxplot(x=col_a, y=col_b, data=data, whis=2.0)
    plt.show()

