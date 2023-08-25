class Prestacion_definida:

    def lx_participes():

        import pandas as pd

        raw1 = 'https://raw.githubusercontent.com/Joevalencia/Planes_pensione_ALM/main/Participes_aportacion.csv'
        data1 = pd.read_csv(raw1).rename(columns={'Antoni': 'Joan'})
        raw2 = 'https://raw.githubusercontent.com/Joevalencia/Planes_pensione_ALM/main/approximation_planex.csv'
        data2 = pd.read_csv(raw2).rename(columns={'Antoni': 'Joan'})
        del data1['Unnamed: 0']
        del data2['Unnamed: 0']
        return data1, data2

    def variation_capital():
        import pandas as pd
        j = 'https://raw.githubusercontent.com/Joevalencia/Planes_pensione_ALM/main/variation6.csv'
        ok = pd.read_csv(j)
        k = 'https://raw.githubusercontent.com/Joevalencia/Planes_pensione_ALM/main/px_mens_participes.csv'
        ok1 = pd.read_csv(k)
        return ok, ok1