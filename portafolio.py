"""
Portfolio Management - Coursework 2: Asset simulation and Liability Driven Portfolios
Welcome to Coursework 2! In this coursework you will dive deeper into portfolio construction,this time using the concepts of asset simulation and liability-driven portfolio management. Different to cw1, this implies start thinking about investment strategies.

This assignment is designed to practice all the concepts studied in this section of the course, and it is mainly build upon your foundational knowledge in mathematical finance, with a strong focus on practical application and critical analysis of portfolio management strategies. In particular, we will explore how to simulate asset returns, construct liability-driven portfolios, and analyze the alignment between asset performance and liability obligations.

Here's a breakdown of the topics you will cover:

Portfolio Management Basics:

We use the foundational concepts in managing a collection of financial assets studied in the first part of the course, and in your previous courses in mathematical finance. The primary goal is to maximize returns while managing risks through diversification, asset allocation, and selection strategies.

Asset Simulation: You will explore asset simulation techniques, which are essential for predicting future asset values based on historical data, statistical models, or random processes. By simulating possible future outcomes, you will learn to assess potential portfolio performance under different scenarios.

Liability-Driven Investment (LDI): You will be introduced to strategies where investments are selected to meet specific future liabilities, such as pension payouts. LDI focuses on aligning assets with liabilities to minimize the risk of not meeting obligations. This approach is particularly relevant for institutions like pension funds, where cash flow alignment is crucial.

There are two types of exercises, those marked with letter E represent programmatic exercises that guide you to implement the concepts seen in class. And those marked with R, which are applications or extensions of the basic concepts studied in exercises E and are intended to be used for your report. Thus, your report must be based on exercises R, and to complete exercises R you need to complete exercises E.

Also, this coursework is structured in a progressive fashion, i.e. we use the results from each section for the next one, so that we end up combining all we see through the exercises. As such, you are also expected to write your report in this way, building up a general topic step by step, like telling a story. Try to build up progressively adding cocnepts and results.

Notice that the output of some E exercises is given so that you can check that your implementation is working as expected.

You are expected to:

Complete both types of exercises and write a report with your results and conclusions. If you have done your previous coursework many exercises should be easy to complete.

For the report you must include all the results from exercises R in a story-telling fashion. You need to explain what you are doing, why and how, and to interpret your results and describe what they mean the best you can and how they are related to other results and the general topic.The report is based in exercises R, and to do exercises R you need first to complete exercises E.

No code should be included in the main text of the final report, just include figures, tables, numbers, and your analysis of results and interpretations.

Figures and tables are required to have captions to understand what they represent, and when your refer to them on your report you need to indicate which number of Table, Figure, etc. you are refering to.

The notebook you have obtained with your code must be saved in pdf format and included as a separe file that serves as appendix for the report.

The final report must be less han 3500 words, and contain at the end the count of words. The report must be written in Latex but submitted just as pdf, together with the notebook with your code as pdf. This means that, at the end, you must submit ONLY TWO pdfs.

Any error to meet these conditions will make your work not considered for grading.

You may start now. Good luck!

1 CPPI
You already know how to build a portfolio. Now, we'll implement our first dynamic portfolio strategy called Constant Proportion Portfolio Insurance (CPPI), which balances growth potential with downside protection.

CPPI works by adjusting the portfolio's exposure to risky assets based on a "cushion," which is the difference between the portfolio's current value and a predefined floor (minimum acceptable level). Here's how it functions:

Investment in Risky Asset=Multiplier×(PortfolioValue−Floor)

Define the Floor: Set a minimum value (floor) below which the portfolio should not fall. This could be based on specific needs or risk tolerance.

Calculate the Cushion: The cushion is calculated as the difference between the current portfolio value and the floor. The cushion represents funds that can be invested in risky assets since it’s the amount the portfolio can "afford" to risk.

Apply the Multiplier: A multiplier is used to determine how much of the cushion should be allocated to risky assets. For example, if the multiplier is 3 and the cushion is 100, then 300 would be allocated to risky assets.

Adjust Portfolio Dynamically: As the portfolio's value changes, the cushion changes as well. The portfolio is rebalanced periodically to adjust the allocation between risky and safe assets according to the updated cushion.

When the portfolio performs well, the cushion grows, allowing for more investment in risky assets. Conversely, if the portfolio value falls, the cushion shrinks, prompting a shift toward safer assets to protect the floor. This approach maintains a balance between growth and protection, making CPPI a suitable strategy for investors seeking controlled risk exposure with a guaranteed minimum value.

Load data and libraries
As always strat by loading some libraries and data that would be useful for the practice.
"""
# Importar bibliotecas
import pandas as pd  # Para manejar datos en formato de tablas y series
import numpy as np   # Para cálculos matemáticos y manejo de matrices
import matplotlib.pyplot as plt  # Para crear gráficos y visualizaciones

# Get the number of firms in the index for each industry (index30_nfirms.csv)
# Cargar los datos del índice total del mercado (index30_size.csv)
df_index_size = pd.read_csv("index30_size.csv", header=0, index_col=0, dtype=str)  # Cargar como string
df_index_size.index = pd.to_datetime(df_index_size.index, format='%Y%m')  # Convertir formato YYYYMM a datetime
df_index_size.index = df_index_size.index.to_period('M')  # Convertir a PeriodIndex mensual
df_index_size.columns = df_index_size.columns.str.strip()  # Eliminar espacios en los nombres de las columnas

"""
E1. Load the total market index size and industry firms. Give time series format in monthly periods. Note: Do not transform to percentages.
"""
# Cargar el número de empresas en el índice para cada industria (index30_nfirms.csv)
df_index_firms = pd.read_csv("index30_nfirms.csv", header=0, index_col=0, dtype=str)  # Cargar como string
df_index_firms.index = pd.to_datetime(df_index_firms.index, format='%Y%m')  # Convertir formato YYYYMM a datetime
df_index_firms.index = df_index_firms.index.to_period('M')  # Convertir a PeriodIndex mensual
df_index_firms.columns = df_index_firms.columns.str.strip()  # Eliminar espacios en los nombres de las columnas

# Cargar los datos de los rendimientos totales del mercado (index30_returns.csv)
df_returns = pd.read_csv("index30_returns.csv", header=0, index_col=0, dtype=str)  # Cargar como string
df_returns.index = pd.to_datetime(df_returns.index, format='%Y%m')  # Convertir formato YYYYMM a datetime
df_returns.index = df_returns.index.to_period('M')  # Convertir a PeriodIndex mensual
df_returns = df_returns.astype(float) / 100  # Convertir a porcentajes
df_returns.columns = df_returns.columns.str.strip()  # Eliminar espacios en los nombres de las columnas

# Mostrar las primeras filas para verificar
#print(df_index_size.head())
#print(df_index_firms.head())
#print(df_returns.head())

"""
Now, you have the number of firms in industry for this market, and the average firm size for each of these industries at different periods. You will calculate the market capitalization for each industry and determine the overall market return based on weighted capitalization.

Market capitalization represents the total value of each industry. Understanding each industry's market cap helps determine its influence in the overall market. It is computed as follows:

Industry Market Capi=Number of Firms in Industryi∗Average Firm Sizei 

The sum of the Industry Market Capitalization returns the (Total) Market Capitalization, which is the size of the market at each period.

And, by dividing the capitalization of each industry by the market capitalization we obtain the weighted capitalization. Capitalization weight indicates how much each industry contributes to the overall market. This is essential for calculating a weighted market return that reflects each industry's relative importance.

Once you have these quantities, you can compute the total market return as a weighted average of individual industry returns. This is:

Total Market Return=∑i(Capitalization Weighti∗Industry Returni) 

E2. Calculate Market Capitalization and Weighted Market Return
"""
# Convertir los datos de los DataFrames a numéricos
df_index_firms = df_index_firms.astype(float)  # Convertir a tipo float
df_index_size = df_index_size.astype(float)    # Convertir a tipo float

# Cálculo de la capitalización de mercado
ind_mktcap = df_index_firms * df_index_size  # Número de empresas * Tamaño promedio
total_mktcap = ind_mktcap.sum(axis="columns")  # Suma de la capitalización de mercado para obtener el total

# Capitalización ponderada
ind_capweight = ind_mktcap.divide(total_mktcap, axis="rows")  # División por la capitalización total del mercado

# Rendimiento total del mercado (rendimiento ponderado por capitalización)
total_market_return = (ind_capweight * df_returns).sum(axis="columns")  # Suma ponderada de rendimientos por industria

# Mostrar resultados iniciales
#print("Capitalización de Mercado por Industria:\n", ind_mktcap.head())
#print("\nCapitalización Total del Mercado:\n", total_mktcap.head())
#print("\nPeso de Capitalización por Industria:\n", ind_capweight.head())
#print("\nRendimiento Total del Mercado:\n", total_market_return.head())

"""
For the following, consider just the returns for Steel, Finance, and Beer industries from 2000 onwards. These industries will form our risky assets
"""
# Seleccionar los retornos de las industrias Steel, Finance y Beer desde 2000
df_risky = df_returns.loc['2000-01':, ['Steel', 'Fin', 'Beer']]

# Definir un activo seguro con un rendimiento mensual fijo (por ejemplo, 0.2% mensual)
safe_asset_return = 0.002  # 0.2% en formato decimal
df_safe = pd.DataFrame(safe_asset_return, index=df_risky.index, columns=['Safe Asset'])

# Mostrar resultados
print("Rendimientos de activos de riesgo (Steel, Finance, Beer):\n", df_risky.head())
print("\nRendimientos del activo seguro:\n", df_safe.head())

"""
We also need safe assets. For this, we create a dataframe with the same number of returns and assume a monthly fixed safe return of 0.02 annualized. This is, we have a risk-free asset that pays 0.02 per year.
"""

# Asegúrate de que df_risky está definido
df_risky = df_returns.loc['2000-01':, ['Steel', 'Fin', 'Beer']]

# Crear df_safe con una sola columna y el mismo índice que df_risky
df_safe = pd.DataFrame(0.02 / 12, index=df_risky.index, columns=['Safe Asset'])

# Mostrar resultados
print("\nRendimientos del activo seguro:\n", df_safe.head())

"""
As we have risky and risk-free assets, let's implement CPPI and explore how it works. First, assume we invest 1000 USD, that the floor (the minimum value below which the portfolio should not fall) is 80% of the initial value, and that the multiplier (the level of exposure to risky assets based on the cushion, i.e. the aggressiveness of risky asset allocations) is 3.
"""

# Configurar parámetros iniciales
start = 1000  # Valor inicial de la cuenta en USD
floor = 0.80  # Piso como porcentaje del valor inicial
m = 3  # Multiplicador CPPI

# Inicializar valores
account_value = start  # Valor inicial de la inversión
floor_value = start * floor  # Piso aplicado al valor de la cuenta

# Crear DataFrame para almacenar los resultados
df_portfolio = pd.DataFrame(index=df_risky.index)

# Inicializar las columnas para la inversión en activos de riesgo y activos seguros
df_portfolio['Risky Asset Investment'] = np.nan
df_portfolio['Safe Asset Investment'] = np.nan
df_portfolio['Portfolio Value'] = np.nan

# Asumir que al inicio, toda la inversión está en el activo seguro
df_portfolio.iloc[0, df_portfolio.columns.get_loc('Portfolio Value')] = start
df_portfolio.iloc[0, df_portfolio.columns.get_loc('Risky Asset Investment')] = 0  # Invertir 0 en activos de riesgo
df_portfolio.iloc[0, df_portfolio.columns.get_loc('Safe Asset Investment')] = start  # Invertir todo en activos seguros

# Iterar sobre cada mes para calcular la inversión en activos de riesgo y el valor de la cartera
for i in range(1, len(df_portfolio)):
    # Calcular el cushion (diferencia entre el valor de la cartera y el piso)
    cushion = account_value - floor_value
    
    # Inversión en activos de riesgo
    risky_investment = m * cushion
    
    # Inversión en activos seguros (lo que queda de la cartera)
    safe_investment = account_value - risky_investment
    
    # Actualizar las inversiones en el DataFrame
    df_portfolio.iloc[i, df_portfolio.columns.get_loc('Risky Asset Investment')] = risky_investment
    df_portfolio.iloc[i, df_portfolio.columns.get_loc('Safe Asset Investment')] = safe_investment
    
    # Calcular el rendimiento total del portafolio (suma de la inversión en activos de riesgo y seguro)
    # Rendimiento de los activos de riesgo
    risky_return = (df_risky.iloc[i] * risky_investment).sum()  # Rendimiento ponderado de activos de riesgo
    # Rendimiento del activo seguro
    safe_return = safe_investment * df_safe.iloc[i, 0]
    
    # Calcular el nuevo valor de la cartera
    account_value = risky_investment + risky_return + safe_investment + safe_return
    
    # Actualizar el valor de la cartera en el DataFrame
    df_portfolio.iloc[i, df_portfolio.columns.get_loc('Portfolio Value')] = account_value

# Mostrar resultados
print(df_portfolio.head())


"""
CPPI works through out time, thus we need to define an entity to save the results of our simulated example. For this, we use dataframes of with the same number of trading periods than our risky assets, which are the number of steps for the simulation. As we are interested in tracking the evolution of portfolio values, risky and safe allocations, and total returns over time, we are going to define 3 dataframes.
"""

# Parámetros iniciales
start = 1000  # Valor inicial de la cuenta en USD
floor = 0.80  # Piso como porcentaje del valor inicial
m = 3  # Multiplicador CPPI

# Inicializar valores
account_value = start  # Valor inicial de la inversión
floor_value = start * floor  # Piso aplicado al valor de la cuenta

# Crear DataFrames para hacer seguimiento
dates = df_risky.index
n_steps = len(dates)

# Preparar trackers para el historial de valores de cuenta, cushion y pesos de activos riesgosos
account_history = pd.DataFrame().reindex_like(df_risky)
cushion_history = pd.DataFrame().reindex_like(df_risky)
risky_w_history = pd.DataFrame().reindex_like(df_risky)

# Inicializar la primera fila (al inicio de la simulación)
account_history.iloc[0] = start  # Valor inicial de la cuenta
cushion_history.iloc[0] = account_value - floor_value  # Inicializar cushion
risky_w_history.iloc[0] = 0  # Inicialmente, no hay asignación a activos riesgosos

# Iterar sobre los periodos para realizar la simulación
for i in range(1, n_steps):
    cushion = account_value - floor_value  # Calcular el cushion
    risky_investment = m * cushion  # Inversión en activos riesgosos
    safe_investment = account_value - risky_investment  # Inversión en activos seguros

    # Guardar los valores en los DataFrames de seguimiento
    account_history.iloc[i] = account_value  # Guardar el valor de la cuenta
    cushion_history.iloc[i] = cushion  # Guardar el cushion
    risky_w_history.iloc[i] = risky_investment / account_value  # Guardar el peso de activos riesgosos

    # Calcular el rendimiento de la cartera
    risky_return = (df_risky.iloc[i] * risky_investment).sum()  # Rendimiento de activos riesgosos
    safe_return = safe_investment * df_safe.iloc[i, 0]  # Rendimiento de activos seguros
    account_value = risky_investment + risky_return + safe_investment + safe_return  # Nuevo valor de la cuenta

# Mostrar los resultados
print("Historial de valores de la cuenta:\n", account_history.head())
print(account_history.tail())
print("\nHistorial de cushions:\n", cushion_history.head())
print(cushion_history.tail())
print("\nHistorial de pesos en activos riesgosos:\n", risky_w_history.head())
print(risky_w_history.tail())

"""
Awesome, according to the definition we have everything we need to simulate a CPPI strategy. The following code implements CPPI with a loop that adjusts the portfolio's allocation between risky and safe assets dynamically based on changes in the portfolio value and cushion, aiming to grow the portfolio while maintaining a minimum guaranteed level (floor). This implies:

Calculate the cushion:  Cushion=(Portfolio Value−Floor)/Portfolio Value 
Allocation Weights for the Risky Asset:  Risky Investment Weights=Multiplier×Cushion 
Allocation Weights for the Safe Asset: Any remaining funds are allocated to the safe asset.
Allocate money to assets: Apply the weights to the current account value.
Update the account value.
New Portfolio Value=(Allocation to Risky×(1+Risky Return))+(Allocation to Safe×(1+Safe Return)) 

Remember, our defined parameters allow you to control the level of risk and protection within the CPPI strategy. The floor ensures downside protection, while the multiplier allows growth by dynamically allocating more to risky assets when the portfolio performs well.

Caution: The loop saves the values being tracket to the dfs we defined to track them, so if you run the loop completely it would change the values saved.
"""

# Supongamos que ya tienes los DataFrames df_risky y df_safe
# df_risky debe tener columnas como 'Steel', 'Fin', 'Beer'
# df_safe debe contener los rendimientos de los activos seguros.

# Parámetros iniciales
start = 1000  # Valor inicial de la cuenta en USD
floor = 0.80  # Piso como porcentaje del valor inicial
m = 3  # Multiplicador CPPI

# Inicializar valores
account_value = start  # Valor inicial de la inversión
floor_value = start * floor  # Piso aplicado al valor de la cuenta

# Crear DataFrames para hacer seguimiento
dates = df_risky.index
n_steps = len(dates)

# Preparar trackers para el historial de valores de cuenta, cushion y pesos de activos riesgosos
account_history = pd.DataFrame(index=dates, columns=['Steel', 'Fin', 'Beer'])
cushion_history = pd.DataFrame(index=dates, columns=['Steel', 'Fin', 'Beer'])
risky_w_history = pd.DataFrame(index=dates, columns=['Steel', 'Fin', 'Beer'])

# Inicializar la primera fila (al inicio de la simulación)
account_history.iloc[0] = start  # Valor inicial de la cuenta
cushion_history.iloc[0] = account_value - floor_value  # Inicializar cushion
risky_w_history.iloc[0] = 0  # Inicialmente, no hay asignación a activos riesgosos

# Iterar sobre los periodos para realizar la simulación
for step in range(1, n_steps):
    # Calcular el cushion
    cushion = (account_value - floor_value) / account_value

    # Calcular el peso para la asignación a activos riesgosos
    risky_w = m * cushion

    # Asegurar que la asignación a activos riesgosos no supere el 100% de la cuenta ni sea negativa
    risky_w = np.minimum(risky_w, 1)
    risky_w = np.maximum(risky_w, 0)

    # Calcular el peso para los activos seguros (el resto de la cuenta se asigna a seguros)
    safe_w = 1 - risky_w

    # Asignar dinero a los activos riesgosos y seguros
    risky_alloc = risky_w * account_value
    safe_alloc = safe_w * account_value

    # Actualizar el valor de la cuenta según los rendimientos de los activos riesgosos y seguros
    risky_return = (df_risky.iloc[step] * risky_alloc).sum()  # Rendimiento de activos riesgosos
    safe_return = safe_alloc * df_safe.iloc[step]  # Rendimiento de activos seguros
    account_value = risky_alloc + risky_return + safe_alloc + safe_return  # Nuevo valor de la cuenta

    # Guardar los valores en los DataFrames de seguimiento
    cushion_history.iloc[step] = cushion
    account_history.iloc[step] = account_value
    risky_w_history.iloc[step] = risky_w

# Mostrar los resultados
print("Historial de valores de la cuenta:\n", account_history.head())
print(account_history.tail())
print("\nHistorial de cushions:\n", cushion_history.head())
print(cushion_history.tail())
print("\nHistorial de pesos en activos riesgosos:\n", risky_w_history.head())
print(risky_w_history.tail())

#First values of our CCPI
account_history.head()

"""
Before seeing the effects of CPPI strategy, what would have happened if we had put all the money in the risky assets and not using the CPPI? Well, this is basically the cumulative returns of the risky assets:
"""

#Plot the account history for one asset, comparing CPPI-managed wealth with a fully risky allocation strategy.
risky_wealth = start * (1 + df_risky).cumprod()
risky_wealth.plot()

"""
But, what is the investment allocation recommended using CPPI? Well, we can know this by plotting our simulated weights.
"""

risky_w_history.plot()

"""
This is the evolution of the allocation to risky assets. Notice the increment in investment on beer. Let's compare then the CPPI vs Full Risky Allocation to beer:
"""

import matplotlib.pyplot as plt

# Simulación de CPPI para Beer
# Ya tienes los datos de account_history["Beer"], risky_wealth["Beer"]
# Es importante que risky_wealth ya esté calculado como el crecimiento de la inversión total en activos riesgosos.

# Evolución de la riqueza de CPPI para Beer
ax = account_history["Beer"].plot(figsize=(12, 6), title="CPPI vs Full Risky Allocation")

# Evolución de la riqueza de Full Risky Allocation para Beer
risky_wealth["Beer"].plot(ax=ax, style="k:", label="Full Risky Allocation (Beer)")

# Añadir la línea del valor del piso
plt.axhline(y=floor_value, color='r', linestyle="--", label="Floor Value")  # Línea de piso

# Mostrar leyenda
plt.legend()

# Mostrar el gráfico
plt.show()

"""
What can you observe?

Notice that allocation to beer increased, and we are far away from the floor. The no CPPI had more volatility. In practice, we would had incresed the floor, as we are far away from it, otherwise we are very conservative.

E4. Compare CPPI vs Risky Allocation for Finance and Steel
"""

# Plot CPPI-managed wealth vs. full-risky strategy for Fin (Finance)
ax = account_history["Fin"].plot(figsize=(12, 6), title="CPPI vs Full Risky Allocation (Finance)")
risky_wealth["Fin"].plot(ax=ax, style="k:", label="Full Risky Allocation (Finance)")
plt.axhline(y=floor_value, color='r', linestyle="--", label="Floor Value")  # Plot the floor line
plt.legend()
plt.show()


"""
Here the effect of CPPI is more cleared, as in 2009 the allocation was really good, we had no violation and we had protection when the market crashed, the defect was that when the market rose we didn't enjoy all the upside benefit.
"""

# Plot CPPI-managed wealth vs. full-risky strategy for Steel
ax = account_history["Steel"].plot(figsize=(12, 6), title="CPPI vs Full Risky Allocation (Steel)")
risky_wealth["Steel"].plot(ax=ax, style="k:", label="Full Risky Allocation (Steel)")
plt.axhline(y=floor_value, color='r', linestyle="--", label="Floor Value")  # Plot the floor line
plt.legend()
plt.show()

"""

To further analyze this, we are going to make use of our previous risk-returns metrics from cw1. Namely, annualized returns, annualized volatility, skewness, kurtosis, Cornish-Fisher VaR, Historic VaR, CVaR, Sharpe Ratio, and Max Drawdown.

E5. Compute the summary statistics studied in CW1 and apply them to df_risky
"""

def summary_stats(r):
    """
    Calculate summary statistics for returns.
    Includes annualized return, volatility, skewness, kurtosis, VaR, CVaR, Sharpe ratio, and max drawdown.
    """
    # Calculate annualized return using compounded growth
    compounded_growth = (1 + r).prod()
    n_periods = r.shape[0]
    ann_r = (compounded_growth ** (252 / n_periods)) - 1  # 252 días de trading al año

    # Annualized volatility (252 días de trading al año)
    ann_vol = r.std() * np.sqrt(252)

    # Skewness and kurtosis
    skew = r.skew()
    kurt = r.kurt()

    # Value at Risk (5%) using Cornish-Fisher expansion
    z = 1.645  # Z-score for 5% VaR
    cf_var5 = (r.mean() - z * r.std())

    # Historical VaR (5%)
    hist_var5 = -r.quantile(0.05)

    # Conditional Value at Risk (CVaR)
    cvar5 = r[r <= hist_var5].mean()

    # Sharpe Ratio assuming a risk-free rate of 0 for simplicity
    sharpe_ratio = ann_r / ann_vol.replace(0, np.nan)  # Avoid division by zero by replacing 0 with NaN

    # Maximum drawdown
    cumulative = (1 + r).cumprod()
    peak = cumulative.cummax()
    max_dd = ((cumulative / peak) - 1).min()

    # Compile results into a DataFrame
    return pd.DataFrame({
        "Annualized Return": ann_r,
        "Annualized Vol": ann_vol,
        "Skewness": skew,
        "Kurtosis": kurt,
        "Cornish-Fisher VaR(5%)": cf_var5,
        "Historic VaR(5%)": hist_var5,
        "CVar(5%)": cvar5,
        "Sharpe Ratio": sharpe_ratio,
        "Max Drawdown": max_dd
    })

# Display summary statistics for risky assets
summary_stats(df_risky)

#Note: The values for Steel are:
#Annualized Return	Annualized Vol	Skewness	Kurtosis	Cornish-Fisher VaR(5%)	Historic VaR(5%)	CVar(5%)	Sharpe Ratio	Max Drawdown
#Steel	-0.002790	0.312368	-0.328499	1.196664	-0.480687	0.140580	0.208117	-0.008931	-0.758017


"""
Great. You have already implemented CPPI and metrics to evaluate investment strategies. It is time to transform our strategy into a function that we can use whenever we want.

E6. Implement CPPI as a function
"""

def run_cppi(risky_r, safe_r=None, m=3, start=1000, floor=0.8, riskfree_rate=0.03):
    """
    Returns a basket of the CPPI strategy, given a set of returns for the risky asset.
    Returns a dictionary containing: Asset Value History, Risk Budget History, Risky weight history.
    """
    # Set up the parameters
    dates = risky_r.index
    n_steps = len(risky_r)
    account_value = start
    floor_value = start * floor

    # Check if we are given correct data
    if isinstance(risky_r, pd.Series):
        risky_r = pd.DataFrame(risky_r, columns=["R"])

    # If no safe asset returns are provided, assume a constant risk-free rate
    if safe_r is None:
        safe_r = pd.DataFrame(riskfree_rate / 12, index=risky_r.index, columns=["R"])

    # Set up DataFrames to track the evolution of variables
    account_history = pd.DataFrame(index=dates)
    cushion_history = pd.DataFrame(index=dates)
    risky_w_history = pd.DataFrame(index=dates)

    # CPPI implementation
    for step in range(n_steps):
        # Calculate the cushion (wealth above the floor)
        cushion = account_value - floor_value

        # Calculate the weight for risky assets (proportional to the cushion)
        risky_w = m * cushion / account_value
        risky_w = np.maximum(risky_w, 0)  # No short selling

        # Allocation to risky and safe assets
        risky_alloc = risky_w * account_value
        safe_alloc = account_value - risky_alloc

        # Update the account value: risky asset return * risky allocation + safe asset return * safe allocation
        account_value = risky_alloc * (1 + risky_r.iloc[step]["R"]) + safe_alloc * (1 + safe_r.iloc[step]["R"])

        # Store the results for this step
        cushion_history.iloc[step] = cushion
        account_history.iloc[step] = account_value
        risky_w_history.iloc[step] = risky_w

    # Calculate risky wealth (wealth from risky assets)
    risky_wealth = start * (1 + risky_r).cumprod()

    # Pack all our backtest info into a dictionary
    backtest_result = {
        "Wealth": account_history,
        "Risky Wealth": risky_wealth,
        "Risk Budget": cushion_history,
        "Risky Allocation": risky_w_history,
        "m": m,
        "start": start,
        "floor": floor,
        "risky_r": risky_r,
        "safe_r": safe_r
    }
    
    return backtest_result
"""
Apply CCPI to our df_risky and compute the summary statistics.
"""