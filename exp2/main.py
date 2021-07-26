from pathlib import Path
import pandas as pd
import numpy as np
from matplotlib  import pyplot as plt


df = pd.DataFrame({
    "a": range(10),
    "b": range(10)
})


r = df.groupby("a").sum()

print(r)

plt.plot(df["b"].values**2)
plt.title("your charts")
plt.show()

plt.plot(df["a"]**2 * df["b"]**3)
plt.title("my charts")
plt.show()