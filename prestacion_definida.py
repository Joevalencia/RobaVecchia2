def prestacion_definida(df, fecha_valoracion='31/12/2021', rvs=.015, rvp=.01, h=12, margen_sol=.02,
                        tipo_int=.0022, beta=.15, edad_jubilacion=65):
    import pandas as pd
    import numpy as np
    import planex
    from pandas.core.common import SettingWithCopyWarning
    import warnings
    warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
    datos = df

    # def lx_participes():
    #
    #     raw1 = 'https://raw.githubusercontent.com/Joevalencia/Planes_pensione_ALM/main/Participes_aportacion.csv'
    #     data1 = pd.read_csv(raw1).rename(columns={'Antoni': 'Joan'})
    #     raw2 = 'https://raw.githubusercontent.com/Joevalencia/Planes_pensione_ALM/main/approximation_planex.csv'
    #     data2 = pd.read_csv(raw2).rename(columns={'Antoni': 'Joan'})
    #     del data1['Unnamed: 0']
    #     del data2['Unnamed: 0']
    #     return data1, data2
    #
    # data1 = lx_participes()[0]
    # data2 = lx_participes()[1]
    year = int(fecha_valoracion.split('/')[2])

    def lx_participes():
        return planex.Prestacion_definida().lx_participes()

    data1 = lx_participes()[0]
    data2 = lx_participes()[1]
    #         p1 = str(input(f'Poner el nombre del participe del Plan numero {i+1}: '))
    #         p2 = str(input(f'Ingresar la fecha de nacimiento {p1}: '))
    #         p3 = str(input(f'Ingresar la fecha de alta en la empresa de {p1}: '))
    #         p4 = float(input(f'Ingresar el salario pensionable 2021 de {p1}: '))
    #         p5 = str(input(f'Ingresar el Sexo de {p1}: '))
    #         print('='*55)
    #         l1.append(p1)
    #         l2.append(p2)
    #         l3.append(p3)
    #         l4.append(p4)
    #         l5.append(p5)

    array1 = []
    for i in range(0, len(df['Participe'])):
        array1.append(int(df['Nacimiento'][i].split('/')[2]))

    array2 = []
    for i in range(0, len(df['Participe'])):
        array2.append(int(df['Alta'][i].split('/')[2]))
    an1 = [46, 38, 63, 30, 30, 24]  ## edad actuarial  # AAA poner manualmente poner esto
    an2 = [25, 37, 29, 26, 20, 23]  ## edad entradad actuarial  # AAA poner manualmente esto
    edad = [x - y for (x, y) in zip(array2, array1)]  ## edad actuarial 2
    r1 = 1 + rvs  ## Revalorizaciòn salarios
    r2 = 1 + rvp  ## Revalorizaciòn pensiones
    interes = (1 + tipo_int)  ##interés
    t1 = np.arange(0, ((120 - edad_jubilacion) * h))  ## mensilidad
    interes3 = interes ** -(t1 / h)  ## fraccionamiento 1/h

    # interes3 = interes2**-t1  ## intereses mensuales
    # def variation_capital():
    #     j = 'https://raw.githubusercontent.com/Joevalencia/Planes_pensione_ALM/main/variation6.csv'
    #     ok = pd.read_csv(j)
    #     k = 'https://raw.githubusercontent.com/Joevalencia/Planes_pensione_ALM/main/px_mens_participes.csv'
    #     ok1 = pd.read_csv(k)
    #     return ok, ok1

    def variation_capital():
        return planex.Prestacion_definida().variation_capital()
    hello = variation_capital()
    ok, ok1 = hello[0], hello[1]
    #     df = pd.DataFrame({'Participe':pd.Series(l1), 'Sexo': pd.Series(l5),'Nacimiento':pd.Series(l2),
    #                        'Alta':pd.Series(l3), 'Salario 2021':pd.Series(l4),
    #                       'Edad actuarial':pd.Series(an1), 'Edad entrada':pd.Series(an2)})

    e1 = round(((pd.to_datetime(datos['Alta']) - pd.to_datetime(datos['Nacimiento'])) / 365.25) / np.timedelta64(1,
                                                                                                                 'D'))  ## Edad E
    fec_val = []
    for i in range(0, 6):
        fec_val.append(fecha_valoracion)
    fec_val = pd.Series(fec_val)
    act_edad = round(
        ((pd.to_datetime(fec_val) - pd.to_datetime(datos['Nacimiento'])) / 365.25) / np.timedelta64(1, 'D'))  ## Edad A
    df = df.join(pd.Series(act_edad, name='Edad actuarial'))
    df = df.join(pd.Series(e1, name='Edad entrada'))
    jubi2 = pd.to_numeric(df['Nacimiento'].str.split('/', expand=True)[2]) + edad_jubilacion
    mes = pd.to_numeric(df['Nacimiento'].str.split('/', expand=True)[1])
    df = df.join(pd.Series(jubi2, name='Jubilacion'))
    proyec = round(df['Salario 2021'] * (r1) ** (edad_jubilacion - 1 - df['Edad actuarial']), 2)
    pj = round(beta * proyec / h, 2)
    df['Proyectado'] = proyec
    df['Proyectado'][2] = datos['Salario 2021'][2]

    jubi3 = pd.to_numeric(df['Alta'].str.split('/', expand=True)[2])
    df['PJ corriente'] = pj
    df['PJ corriente'][2] = beta * df['Proyectado'][2] / h
    joan1 = ok['Joan_var'] * (
        (data2.Joan[edad_jubilacion * h + mes[0] + 1:] / data2.Joan[edad_jubilacion * h + mes[0]]).reset_index(
            drop=True))
    pere1 = ok['Pere_var'] * (
        (data2.Pere[edad_jubilacion * h + mes[1] + 1:] / data2.Pere[edad_jubilacion * h + mes[1]]).reset_index(
            drop=True))
    marta1 = ok['Marta_var'] * (
        (data2.Marta[edad_jubilacion * h + mes[2] + 1:] / data2.Marta[edad_jubilacion * h + mes[2]]).reset_index(
            drop=True))
    claudia1 = ok['Claudia_var'] * (
        (data2.Claudia[edad_jubilacion * h + mes[3] + 1:] / data2.Claudia[edad_jubilacion * h + mes[3]]).reset_index(
            drop=True))
    marc1 = ok['Marc_var'] * (
        (data2.Marc[edad_jubilacion * h + mes[4] + 1:] / data2.Marc[edad_jubilacion * h + mes[4]]).reset_index(
            drop=True))
    maria1 = ok['Maria_var'] * (
        (data2.Maria[edad_jubilacion * h + mes[5] + 1:] / data2.Maria[edad_jubilacion * h + mes[5]]).reset_index(
            drop=True))
    new_df = pd.DataFrame({'Joan_VAA': joan1 * interes3, 'Pere_VAA': pere1 * interes3, 'Marta_VAA': marta1 * interes3,
                           'Claudia_VAA': claudia1 * interes3, 'Marc_VAA': marc1 * interes3,
                           'Maria_VAA': maria1 * interes3})
    VAAs = pd.Series(new_df.sum().round(2).to_list(), name='VAAj')

    ## A la fecha actual 2021
    VAA21_joan = round(VAAs[0] * ((1 + tipo_int) ** -(edad_jubilacion - df['Edad actuarial'][0])) * (
            data1['Joan'][edad_jubilacion] / data1['Joan'][df['Edad actuarial'][0]]), 2)
    VAA21_pere = round(VAAs[1] * ((1 + tipo_int) ** -(edad_jubilacion - df['Edad actuarial'][1])) * (
            data1['Pere'][edad_jubilacion] / data1['Pere'][df['Edad actuarial'][1]]), 2)
    VAA21_marta = round(VAAs[2] * ((1 + tipo_int) ** -(edad_jubilacion - df['Edad actuarial'][2])) * (
            data1['Marta'][edad_jubilacion] / data1['Marta'][df['Edad actuarial'][2]]), 2)
    VAA21_claudia = round(VAAs[3] * ((1 + tipo_int) ** -(edad_jubilacion - df['Edad actuarial'][3])) * (
            data1['Claudia'][edad_jubilacion] / data1['Claudia'][df['Edad actuarial'][3]]), 2)
    VAA21_marc = round(VAAs[4] * ((1 + tipo_int) ** -(edad_jubilacion - df['Edad actuarial'][4])) * (
            data1['Marc'][edad_jubilacion] / data1['Marc'][df['Edad actuarial'][4]]), 2)
    VAA21_maria = round(VAAs[5] * ((1 + tipo_int) ** -(edad_jubilacion - df['Edad actuarial'][5])) * (
            data1['Maria'][edad_jubilacion] / data1['Maria'][df['Edad actuarial'][5]]), 2)
    vaas21 = pd.Series([VAA21_joan, VAA21_pere, VAA21_marta, VAA21_claudia, VAA21_marc, VAA21_maria])

    ## A la fecha de alta en la empresa
    VAAp_joan = round(VAAs[0] * ((1 + tipo_int) ** -(jubi2[0] - jubi3[0] + 1)) * (
            data1['Joan'][edad_jubilacion] / data1['Joan'][df['Edad entrada'][0]]), 2)
    VAAp_pere = round(VAAs[1] * ((1 + tipo_int) ** -(jubi2[1] - jubi3[1] + 1)) * (
            data1['Pere'][edad_jubilacion] / data1['Pere'][df['Edad entrada'][1]]), 2)
    VAAp_marta = round(VAAs[2] * ((1 + tipo_int) ** -(jubi2[2] - jubi3[2] + 1)) * (
            data1['Marta'][edad_jubilacion] / data1['Marta'][df['Edad entrada'][2]]), 2)
    VAAp_claudia = round(VAAs[3] * ((1 + tipo_int) ** -(jubi2[3] - jubi3[3] + 1)) * (
            data1['Claudia'][edad_jubilacion] / data1['Claudia'][df['Edad entrada'][3]]), 2)
    VAAp_marc = round(VAAs[4] * ((1 + tipo_int) ** -(jubi2[4] - jubi3[4] + 1)) * (
            data1['Marc'][edad_jubilacion] / data1['Marc'][df['Edad entrada'][4]]), 2)
    VAAp_maria = round(VAAs[5] * ((1 + tipo_int) ** -(jubi2[5] - jubi3[5] + 1)) * (
            data1['Maria'][edad_jubilacion] / data1['Maria'][df['Edad entrada'][5]]), 2)
    vaas_p = pd.Series([VAAp_joan, VAAp_pere, VAAp_marta, VAAp_claudia, VAAp_marc, VAAp_maria])

    def coste_normal_cunitario():

        vl1 = VAAs[0] * ((1 + tipo_int) ** -np.flip(np.arange(0, (edad_jubilacion-df['Edad entrada'][0]))))
        vl1 = round(vl1 * (data1['Joan'][edad_jubilacion] / data1['Joan'][np.arange(df['Edad entrada'][0], edad_jubilacion)]),2)

        vl2 = VAAs[1] * ((1 + tipo_int) ** -np.flip(np.arange(df['Edad entrada'][1], edad_jubilacion)))
        vl2 = round(vl2 * (data1['Pere'][edad_jubilacion] / data1['Pere'][np.arange(df['Edad entrada'][1], edad_jubilacion)]),2)

        vl3 = round(VAAs[2] * ((1 + tipo_int) ** -np.flip(np.arange(df['Edad entrada'][2], edad_jubilacion))) * (
                data1['Marta'][edad_jubilacion] / data1['Marta'][np.arange(df['Edad entrada'][2], edad_jubilacion)]), 2)

        vl4 = round(VAAs[3] * ((1 + tipo_int) ** -np.flip(np.arange(df['Edad entrada'][3], edad_jubilacion))) * (
                data1['Claudia'][edad_jubilacion] / data1['Claudia'][np.arange(df['Edad entrada'][3], edad_jubilacion)]), 2)

        vl5 = round(VAAs[4] * ((1 + tipo_int) ** -np.flip(np.arange(df['Edad entrada'][4], edad_jubilacion))) * (
                data1['Marc'][edad_jubilacion] / data1['Marc'][np.arange(df['Edad entrada'][4], edad_jubilacion)]), 2)

        vl6 = round(VAAs[5] * ((1 + tipo_int) ** -np.flip(np.arange(df['Edad entrada'][5], edad_jubilacion))) * (
                data1['Maria'][edad_jubilacion] / data1['Maria'][np.arange(df['Edad entrada'][5], edad_jubilacion)]), 2)

        # cn_joan = vl1[int(df['Edad actuarial'][0]):] / np.flip(np.arange(df['Edad entrada'][0], edad_jubilacion)#[int(df['Edad actuarial'][0] - df['Edad entrada'][0]):])
        # cn_pere = vl2[int(df['Edad actuarial'][1]):] / np.flip(np.arange(df['Edad entrada'][1], edad_jubilacion)[int(df['Edad actuarial'][1] - df['Edad entrada'][1]):])
        # cn_marta = vl3[int(df['Edad actuarial'][2]):] / np.flip(np.arange(df['Edad entrada'][2], edad_jubilacion)[int(df['Edad actuarial'][2] - df['Edad entrada'][2]):])
        # cn_claudia = vl4[int(df['Edad actuarial'][3]):] / np.flip(np.arange(df['Edad entrada'][3], edad_jubilacion)[int(df['Edad actuarial'][3] - df['Edad entrada'][3]):])
        # cn_marc = vl5[int(df['Edad actuarial'][4]):] / np.flip(np.arange(df['Edad entrada'][4], edad_jubilacion)[int(df['Edad actuarial'][4] - df['Edad entrada'][4]):])
        # cn_maria = vl6[int(df['Edad actuarial'][5]):] / np.flip(np.arange(df['Edad entrada'][5], edad_jubilacion)[int(df['Edad actuarial'][5] - df['Edad entrada'][5]):])

        return vl1, vl2, vl3, vl4, vl5, vl6
        #vaas_p = pd.Series([VAAp_joan, VAAp_pere, VAAp_marta, VAAp_claudia, VAAp_marc, VAAp_maria])


    ## Coste normal Credito Unitario
    jo1 = vaas21[0] / (edad_jubilacion - df['Edad entrada'][0])
    jo2 = vaas21[1] / (edad_jubilacion - df['Edad entrada'][1])
    jo3 = vaas21[2] / (edad_jubilacion - df['Edad entrada'][2])
    jo4 = vaas21[3] / (edad_jubilacion - df['Edad entrada'][3])
    jo5 = vaas21[4] / (edad_jubilacion - df['Edad entrada'][4])
    jo6 = vaas21[5] / (edad_jubilacion - df['Edad entrada'][5])
    CN_credito = pd.Series([jo1, jo2, jo3, jo4, jo5, jo6])

    ## Valor actual actuarial servicios pasados
    VAASP_CU = pd.Series([CN_credito[0] * (df['Edad actuarial'][0] - df['Edad entrada'][0]),
                          CN_credito[1] * (df['Edad actuarial'][1] - df['Edad entrada'][1]),
                          CN_credito[2] * (df['Edad actuarial'][2] - df['Edad entrada'][2]),
                          CN_credito[3] * (df['Edad actuarial'][3] - df['Edad entrada'][3]),
                          CN_credito[4] * (df['Edad actuarial'][4] - df['Edad entrada'][4]),
                          CN_credito[5] * (df['Edad actuarial'][5] - df['Edad entrada'][5])])
    ## Margen de solvencia
    margen_CU = VAASP_CU * margen_sol

    ## dataframe Credito Unitario
    CU = pd.DataFrame({'VAAj': VAAs, 'VAA 31/12/2021': vaas21,
                       'CN 31/12/2021': CN_credito, 'VAASP 31/12/2021': VAASP_CU, 'MS 31/12/2021': margen_CU}).round(2)
    CU.index = df['Participe'].to_list()
    CU['VAASF 31/12/2021'] = CU.iloc[:, 1] - CU.iloc[:, 3]

    ## Edad de entrada - Coste Normal a la alta
    tiempo1 = np.arange(0, (edad_jubilacion - df['Edad entrada'][0]))
    # jo_1 =
    jo_1 = vaas_p[0] / sum(((1 + rvs) ** tiempo1 * (1 + tipo_int) ** -(tiempo1)) * (
            data1['Joan'][df['Edad entrada'][0] + tiempo1] / data1['Joan'][df['Edad entrada'][0]]))
    tiempo2 = np.arange(0, (edad_jubilacion - df['Edad entrada'][1]))
    # jo_2 =
    jo_2 = vaas_p[1] / sum(((1 + rvs) ** tiempo2 * (1 + tipo_int) ** -(tiempo2)) * (
            data1['Pere'][df['Edad entrada'][1] + tiempo2] / data1['Pere'][df['Edad entrada'][1]]))
    tiempo3 = np.arange(0, (edad_jubilacion - df['Edad entrada'][2]))
    # jo_3 =
    jo_3 = vaas_p[2] / sum(((1 + rvs) ** tiempo3 * (1 + tipo_int) ** -(tiempo3)) * (
            data1['Marta'][df['Edad entrada'][2] + tiempo3] / data1['Claudia'][df['Edad entrada'][2]]))
    tiempo4 = np.arange(0, (edad_jubilacion - df['Edad entrada'][3]))
    # jo_4 =
    jo_4 = vaas_p[3] / sum(((1 + rvs) ** tiempo4 * (1 + tipo_int) ** -(tiempo4)) * (
            data1['Claudia'][df['Edad entrada'][3] + tiempo4] / data1['Marta'][df['Edad entrada'][3]]))
    tiempo5 = np.arange(0, (edad_jubilacion - df['Edad entrada'][4]))
    # jo_5 =
    jo_5 = vaas_p[4] / sum(((1 + rvs) ** tiempo5 * (1 + tipo_int) ** -(tiempo5)) * (
            data1['Marc'][df['Edad entrada'][4] + tiempo5] / data1['Marc'][df['Edad entrada'][4]]))
    tiempo6 = np.arange(0, (edad_jubilacion - df['Edad entrada'][5]))
    # jo_6 =
    jo_6 = vaas_p[5] / sum(((1 + rvs) ** tiempo6 * (1 + tipo_int) ** -(tiempo6)) * (
            data1['Maria'][df['Edad entrada'][5] + tiempo6] / data1['Maria'][df['Edad entrada'][5]]))
    CN_EE = pd.Series([jo_1, jo_2, jo_3, jo_4, jo_5, jo_6]).round(2)  ## Coste Normal Edad de entrada en el 0

    # Coste Normal en el 2021 de los participes
    p1 = CN_EE[0] * (1 + rvs) ** (df['Edad actuarial'][0] - df['Edad entrada'][0])
    p2 = CN_EE[1] * (1 + rvs) ** (df['Edad actuarial'][1] - df['Edad entrada'][1])
    p3 = CN_EE[2] * (1 + rvs) ** (df['Edad actuarial'][2] - df['Edad entrada'][2])
    p4 = CN_EE[3] * (1 + rvs) ** (df['Edad actuarial'][3] - df['Edad entrada'][3])
    p5 = CN_EE[4] * (1 + rvs) ** (df['Edad actuarial'][4] - df['Edad entrada'][4])
    p6 = CN_EE[5] * (1 + rvs) ** (df['Edad actuarial'][5] - df['Edad entrada'][5])

    CN_EE21 = pd.Series([p1, p2, p3, p4, p5, p6]).round(2)

    ## Coste normal futuro

    # a0 = ((1 + rvs) ** (np.arange(0, (edad_jubilacion - df['Edad entrada'][0]))))[
    #      df['Edad actuarial'][0] - df['Edad entrada'][0]:] * CN_EE[0]
    # a1 = ((1 + rvs) ** (np.arange(0, (edad_jubilacion - df['Edad entrada'][1]))))[
    #      df['Edad actuarial'][1] - df['Edad entrada'][1]:] * CN_EE[1]
    # a2 = ((1 + rvs) ** (np.arange(0, (edad_jubilacion - df['Edad entrada'][2]))))[
    #      df['Edad actuarial'][2] - df['Edad entrada'][2]:] * CN_EE[2]
    # a3 = ((1 + rvs) ** (np.arange(0, (edad_jubilacion - df['Edad entrada'][3]))))[
    #      df['Edad actuarial'][3] - df['Edad entrada'][3]:] * CN_EE[3]
    # a4 = ((1 + rvs) ** (np.arange(0, (edad_jubilacion - df['Edad entrada'][4]))))[
    #      df['Edad actuarial'][4] - df['Edad entrada'][4]:] * CN_EE[4]
    # a5 = ((1 + rvs) ** (np.arange(0, (edad_jubilacion - df['Edad entrada'][5]))))[
    #      df['Edad actuarial'][5] - df['Edad entrada'][5]:] * CN_EE[5]

    # Valor actual actuarial servicios pasados edad de entradad al 2021
    w1 = np.arange(0, (edad_jubilacion - df['Edad actuarial'][0]))
    joa1 = round(sum(CN_EE21[0] * (1 + rvs) ** w1 * (1 + tipo_int) ** -w1 * (
            data1['Joan'][df['Edad actuarial'][0] + w1] / data1['Joan'][df['Edad actuarial'][0]])), 2)

    w1 = np.arange(0, (edad_jubilacion - df['Edad actuarial'][1]))
    joa2 = round(sum(CN_EE21[0] * (1 + rvs) ** w1 * (1 + tipo_int) ** -w1 * (
            data1['Joan'][df['Edad actuarial'][1] + w1] / data1['Joan'][df['Edad actuarial'][1]])), 2)

    w1 = np.arange(0, (edad_jubilacion - df['Edad actuarial'][2]))
    joa3 = round(sum(CN_EE21[0] * (1 + rvs) ** w1 * (1 + tipo_int) ** -w1 * (
            data1['Joan'][df['Edad actuarial'][2] + w1] / data1['Joan'][df['Edad actuarial'][2]])), 2)

    w1 = np.arange(0, (edad_jubilacion - df['Edad actuarial'][3]))
    joa4 = round(sum(CN_EE21[0] * (1 + rvs) ** w1 * (1 + tipo_int) ** -w1 * (
            data1['Joan'][df['Edad actuarial'][3] + w1] / data1['Joan'][df['Edad actuarial'][3]])), 2)

    w1 = np.arange(0, (edad_jubilacion - df['Edad actuarial'][4]))
    joa5 = round(sum(CN_EE21[0] * (1 + rvs) ** w1 * (1 + tipo_int) ** -w1 * (
            data1['Joan'][df['Edad actuarial'][4] + w1] / data1['Joan'][df['Edad actuarial'][4]])), 2)

    w1 = np.arange(0, (edad_jubilacion - df['Edad actuarial'][5]))
    joa6 = round(sum(CN_EE21[0] * (1 + rvs) ** w1 * (1 + tipo_int) ** -w1 * (
            data1['Joan'][df['Edad actuarial'][5] + w1] / data1['Joan'][df['Edad actuarial'][5]])), 2)

    serie_vaasp = vaas21 - pd.Series([joa1, joa2, joa3, joa4, joa5, joa6])

    ## Margen metodo de entrada
    mar_EE = serie_vaasp * margen_sol

    df_EE = pd.DataFrame(
        {'VAAj': VAAs, 'VAA 31/12/2021': vaas21, 'CN 31/12/2021': CN_EE21, 'VAASP 31/12/2021': serie_vaasp,
         'VAA en ALTA': vaas_p,
         'MS 31/12/2021': mar_EE.round(2)})
    df_EE['VAASF 31/12/2021'] = df_EE.iloc[:, 1] - df_EE.iloc[:, 3]
    print('=' * 75)
    print('Hipotesis')
    print(f'Interés técnico valoración: {tipo_int * 100}%')
    print(f'Revalorizaciòn Salario: {rvs * 100}%')
    print(f'Revalorización Pensión: {rvp * 100}%')
    print(f'Beta Pension: {beta}')
    print('=' * 75)
    # print(df.set_index('Participe').T)
    # print('=' * 75)
    # print('METODO: Credito Unitario Proyectado')
    # # print(round(CU.T, 2))
    # print(CU)
    # print('=' * 75)
    # print('METODO: Edad de entrada')
    # print(df_EE)

    return df.set_index('Participe'), df_EE, CU, CN_EE, ok
