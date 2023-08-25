def nuevo(cohort1, cohort2, model=None, interactive: bool = False,
          death: bool = False, dx_log: bool = False):
    """
    :param dx_log:
    :param model:
    :param death: Boolean. It display deaths of cohort from Plotly.
    :param cohort1: Array-like Cohort 1
    :param cohort2: Array-like Cohort 2
    :param interactive: Bool: It displays an interactive chart of
     the survival plot using Plotly
    :return: Survival Plot of two cohort
    """

    px1 = ((cohort1 / cohort1.shift(1)).shift(-1)).fillna(0)
    px2 = ((cohort2 / cohort2.shift(1)).shift(-1)).fillna(0)
    qx1, qx2 = 1 - px1, 1 - px2
    dx1, dx2 = qx1 * cohort1, qx2 * cohort2
    e1 = []
    for i in range(0, len(cohort1[:-1])):
        ley = cohort1[i]
        e1.append(sum(cohort1[i + 1:-1]) / ley)
    eyx = pd.Series(e1)
    e2 = []
    for i in range(0, len(cohort2[:-1])):
        ley = cohort2[i]
        e2.append(sum(cohort2[i + 1:-1]) / ley)
    eys = pd.Series(e2)
    df1 = pd.DataFrame({f'{cohort1.name}': cohort1, 'e\u20931': round(eyx, 2),
                        f'{cohort2.name}': cohort2, 'e\u20932': round(eys, 2),
                        'Age': cohort1.index, f'Death {cohort1.name}': dx1,
                        f'Death {cohort2.name}': dx2})

    if not interactive:
        plt.figure(figsize=(15, 5))
        p0 = plt.subplot(1, 2, 1)

        plt.title(f'Survival Plot of {cohort1.name} '
                  f'and {cohort2.name}', fontweight='bold')
        plt.xlabel('Age', fontweight='bold')
        plt.ylabel('Living people ($l_{x}$) and ($l_{y}$)', fontweight='bold')
        plt.xlim([0, 115])
        plt.plot(cohort1.index, cohort1, label=f'{cohort1.name}', color='darkorange')
        plt.plot(cohort2.index, cohort2, label=f'{cohort2.name}', color='midnightblue')
        if model is not None:
            plt.plot(model.index, model, label=f'{model.name}', color='red')
        plt.legend()
        p1 = plt.subplot(1, 2, 2)
        plt.title(f'Death Plot of {cohort1.name} '
                  f'and {cohort2.name}', fontweight='bold')
        plt.ylabel('Death people ($d_{x}$) and ($d_{y}$)', fontweight='bold')
        plt.xlabel('Age', fontweight='bold')
        plt.xlim([0, 115])
        plt.plot(dx1, label=f'{cohort1.name}', color='darkorange')
        plt.plot(dx2, label=f'{cohort2.name}', color='midnightblue')
        if model is not None:
            pxm = ((model / model.shift(1)).shift(-1)).fillna(0)
            qxm = 1 - pxm
            dxm = qxm * model
            plt.plot(dxm, label=f'{model.name}', color='red')
        plt.legend()

        plt.show()

    else:
        import plotly.express as px

        fig = px.line(df1, x='Age', y=[f'{cohort1.name}', f'{cohort2.name}'],
                      hover_data=['e\u20931', 'e\u20932'],
                      title=f'Survival plot of {cohort1.name} and {cohort2.name}')

        if dx_log is not True:
            fig = px.line(df1, x='Age', y=[f'Death {cohort1.name}', f'Death {cohort2.name}'],
                          hover_data=['e\u20931', 'e\u20932'],
                          title=f'Death plot of {cohort1.name} and {cohort2.name}')
        else:
            fig = px.line(df1, x='Age', y=[f'Death {cohort1.name}', f'Death {cohort2.name}'],
                          hover_data=['e\u20931', 'e\u20932'], log_y=True,
                          title=f'Death plot of {cohort1.name} and {cohort2.name}')
        if model is not None:
            pxm = ((model / model.shift(1)).shift(-1)).fillna(0)
            qxm = 1 - pxm
            dxm = qxm * model
            em = []
            for s in range(0, len(model[:-1])):
                luc = model[s]
                em.append(sum(model[s + 1:-1]) / luc)
            emm = pd.Series(em)
            df2 = pd.DataFrame({f'{cohort1.name}': cohort1, 'e\u20931': round(eyx, 2),
                                f'{cohort2.name}': cohort2, 'e\u20932': round(eys, 2),
                                f'{model.name}': model, 'e\u20933': round(emm, 2),
                                'Age': cohort1.index, f'Death {cohort1.name}': dx1,
                                f'Death {cohort2.name}': dx2,
                                f'Death {model.name}': dxm})
            fig = px.line(df2, x='Age', y=[f'Death {cohort1.name}',
                                           f'Death {cohort2.name}',
                                           f'Death {model.name}'],
                          hover_data=['e\u20931', 'e\u20932', 'e\u20933'],
                          title=f'LogDeath plot of {cohort1.name}, {cohort2.name} and {model.name}')
        if dx_log is True:
            fig = px.line(df2, x='Age', y=[f'Death {cohort1.name}',
                                           f'Death {cohort2.name}',
                                           f'Death {model.name}'],
                          hover_data=['e\u20931', 'e\u20932', 'e\u20933'], log_y=True,
                          title=f'LogDeath plot of {cohort1.name}, {cohort2.name} and {model.name}')
        else:

            fig = px.line(df2, x='Age', y=[f'{cohort1.name}',
                                           f'{cohort2.name}',
                                           f'{model.name}'],
                          hover_data=['e\u20931', 'e\u20932', 'e\u20933'],
                          title=f'Survival plot of {cohort1.name}, {cohort2.name} and {model.name}')
        fig.show()
