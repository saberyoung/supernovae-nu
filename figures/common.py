from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/'results'
OUT=ROOT/'figures'/'output'
OUT.mkdir(parents=True,exist_ok=True)

def save(fig,name):
    fig.savefig(OUT/f'{name}.pdf',bbox_inches='tight')
    fig.savefig(OUT/f'{name}.png',dpi=250,bbox_inches='tight')
    plt.close(fig)
