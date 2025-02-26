#!/usr/bin/env python
# coding: utf-8

# In[204]:


import pandas as pd
import numpy as np
import os
from datetime import datetime

# In[ ]:


#os.chdir("C:\\Users\\Ruth Rolón Aranda\\Documents\\Cambra\\Analisis para la consultoria\\ejemplo dash\\Contabilidad")

# In[230]:


# Cargar el archivo CSV
df = pd.read_csv('ingresos_egresos.csv')

# Convertir la columna 'Fecha' en formato datetime
df['Fecha'] = pd.to_datetime(df['Fecha'])

df2 = df.copy()
df2 = df2.loc[df2['Monto']>0]


# Eliminar duplicados en la columna 'Fecha', manteniendo el primer registro
df = df.drop_duplicates(subset=['Fecha'])

# Crear un rango de fechas desde la fecha mínima hasta la fecha máxima
rango_fechas = pd.date_range(start=df['Fecha'].min(), end=df['Fecha'].max(), freq='D')

# Establecer 'Fecha' como índice y reindexar con todas las fechas
df = df.set_index('Fecha').reindex(rango_fechas).fillna(0).reset_index()

# Renombrar la columna 'index' a 'Fecha'
df.rename(columns={'index': 'Fecha'}, inplace=True)

# Copiar el DataFrame
dff = df.copy()

for index, row in dff.iterrows():
    if row['Factura'] == 25:
        dff.at[index, 'Monto'] = 9734700

for index, row in df2.iterrows():
    if row['Factura'] == 25:
        df2.at[index, 'Monto'] = 9734700


# Crear una nueva columna 'Mes' con el formato YYYY-MM
df2['Mes'] = df2['Fecha'].dt.to_period('M').astype(str)




# In[238]:


import dash
from dash import dcc, html, Input, Output, State, Dash
import plotly.express as px
import dash_bootstrap_components as dbc
import dash_table
import plotly.graph_objects as go

# Inicializando la app
app = Dash(external_stylesheets=[dbc.themes.LUX])
server = app.server
fig_1 = {}
fig_2 = {}
fig_3 = {}
fig_4 = {}

# Layout de la aplicación
app.layout = html.Div([
    html.Hr(),

    # Título centrado
    html.Div([
        html.H3("Análisis de Finanzas Personales", style={'text-align': 'center', 'margin': '0'})
    ], style={'display': 'flex', 'align-items': 'center', 'justify-content': 'center'}),

    html.Hr(),

    # Logo centrado debajo del título
    dbc.Row(
        dbc.Col(
            html.Img(src='/assets/CAMBRA.png', style={'height': '60px'}),
            width="auto",  # Esto asegura que la imagen tenga el tamaño adecuado
        ),
        justify="center"  # Asegura que la fila está centrada
    ),

    # DatePickerRange centrado debajo del logo
    dbc.Row(
        dbc.Col(
            dcc.DatePickerRange(
                id='date-picker-range',
                start_date='2023-01-01',
                end_date='2023-01-31',
                display_format='YYYY-MM-DD',
                month_format='MMMM YYYY',
                start_date_placeholder_text='Inicio',
                end_date_placeholder_text='Fin',
            ),
            width="auto",  # Ajuste automático de la columna
        ),
        justify="center",  # Asegura que la fila está centrada
        style={'margin-top': '10px'}
    ),

    html.Hr(),

    # Fila de gráficos
    dbc.Row([
        dbc.Col(dcc.Graph(id='pie-plot', figure=fig_4), md=6),
        dbc.Col(dcc.Graph(id='gastos-plot', figure=fig_2), md=6),
    ]),

    html.Hr(),

    dbc.Row(
        dbc.Col(
            dcc.RadioItems(
                id='radio-options',
                options=[
                    {'label': 'Análisis Diario', 'value': 'diario'},
                    {'label': 'Análisis Mensual' , 'value': 'mensual'},
                ],
                value='diario',
                labelStyle={'display': 'inline-block', 'margin-right': '10px'}
            ),
            width="auto"
            ),
        justify="center",
        style={'margin-top': '10px'},
    ),

    html.Hr(),

    # Fila para el gráfico de pie y la tabla de datos
    dbc.Row([
        dbc.Col(dcc.Graph(id='ingresos-plot', figure=fig_1), md=6),
        dbc.Col(dcc.Graph(id='barra-plot', figure=fig_3), md=6),
    ]),

    # Fila para la tabla de datos
    dbc.Row([
        dbc.Col(
            dash_table.DataTable(
                id='movimientos-table',
                columns=[],
                style_header={'backgroundColor': 'lightblue', 'fontWeight': 'bold'},
                style_table={'height': '300px', 'overflowY': 'auto'},
                style_cell={'textAlign': 'center'},
                data=[],  # Se actualizará vía callback
                filter_action='native',
                sort_action='native',
            ),
            width=12  # Asegúrate de asignar el ancho
        )
    ], style={'margin-top': '10px'})
])

@app.callback(
    [Output('ingresos-plot', 'figure'),
     Output('gastos-plot', 'figure'),
     Output('barra-plot', 'figure'),
     Output('pie-plot', 'figure'),
     Output('movimientos-table', 'data'),
     Output('movimientos-table', 'columns')],
    [Input('radio-options', 'value'),
     Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date')]
)
def update_dashboard( value, start_date, end_date):
    # Filtrar el dataframe según el rango de fechas
    dff_filtrado = dff[(dff['Fecha'] >= start_date) &
                       (dff['Fecha'] <= end_date)]
    df_filtrado = df2[(df2['Fecha'] >= start_date) &
                       (df2['Fecha'] <= end_date)]
        
    dff_ingresos = dff_filtrado.loc[dff_filtrado['Tipo']=='Ingreso'].reset_index()
    dff_egresos = dff_filtrado.loc[dff_filtrado['Tipo']=='Egreso'].reset_index()
    df_ingresos = df_filtrado.loc[df_filtrado['Tipo']=='Ingreso'].reset_index()
    df_egresos = df_filtrado.loc[df_filtrado['Tipo']=='Egreso'].reset_index()
    dff_fig1a = dff_ingresos.groupby(['Fecha', 'Cuenta'])['Monto'].sum().reset_index()
    dff_fig1b = df_ingresos.groupby(['Mes', 'Cuenta'])['Monto'].sum().reset_index()
    dff_fig1b['Fecha'] = dff_fig1b['Mes']
    dff_fig2 = dff_filtrado.groupby(['Tipo', 'Cuenta'])['Monto'].sum().reset_index()
    dff_fig2 = dff_fig2[dff_fig2['Monto'] > 0]
    dff_fig3a = dff_egresos.groupby(['Fecha', 'Cuenta'])['Monto'].sum().reset_index()
    dff_fig3b = df_egresos.groupby(['Mes', 'Cuenta'])['Monto'].sum().reset_index()
    dff_fig3b['Fecha'] = dff_fig3b['Mes']

    dff_tipo = dff_filtrado.groupby(['Tipo'], as_index=False)['Monto'].sum()
    dff_tipo = dff_tipo[dff_tipo['Monto'] > 0]
    dff_cuenta = dff_filtrado.groupby(['Tipo', 'Cuenta'], as_index=False)['Monto'].sum()
    dff_cuenta = dff_cuenta[dff_cuenta['Monto'] > 0]
        # Crear listas para el Sunburst
    labels = []
    parents = []
    values = []

    # Agregar Categorías (Nivel Superior)
    labels.extend(dff_tipo['Tipo'])
    parents.extend([''] * len(dff_tipo['Tipo']))  # Categoría no tiene padre
    values.extend(dff_tipo['Monto'])

    # Agregar Líneas de Productos (Nivel Inferior)
    labels.extend(dff_cuenta['Cuenta'])
    parents.extend(dff_cuenta['Tipo']) 
    values.extend(dff_cuenta['Monto'])
#--------------------------------------------------------------------------------------------------------------------------------------

    radio_options = ['diario', 'mensual']
   
    if value == 'diario':
        dff_fig1 = dff_fig1a
        dff_fig3 = dff_fig3a
    elif value == 'mensual':
        dff_fig1 = dff_fig1b
        dff_fig3 = dff_fig3b 
#--------------------------------------------------------------------------------------------------------------------------------
 
    # Crear figura
    fig_1 = go.Figure()

    # Agregar la venta total
    total_data = dff_fig1.groupby('Fecha')['Monto'].sum().reset_index()
    fig_1.add_trace(go.Scatter(
        x=total_data['Fecha'],
        y=total_data['Monto'],
        mode='lines',
        name='Total',
        line=dict(width=3, color='green')  # Línea más gruesa y verde para diferenciarla
    ))

    # Agregar cada cluster con visibilidad oculta por defecto
    for Cuenta in dff_fig1['Cuenta'].unique():
        cluster_data = dff_fig1[dff_fig1['Cuenta'] == Cuenta]
        fig_1.add_trace(go.Scatter(
            x=cluster_data['Fecha'],
            y=cluster_data['Monto'],
            mode='markers',
            name=str(Cuenta),
            visible="legendonly"  # Hace que los clusters estén ocultos al inicio
        ))

    # Configuración del gráfico
    fig_1.update_layout(
        title="Evolucion de Ingresos",
        xaxis_title='Fecha',
        yaxis_title='Ingresos'
    )


#--------------------------------------------------------------------------------------------------------------------------------
    
    fig_2 = px.bar(dff_fig2, 
            y='Monto',                # Monto comprado por cada cluster
            x='Tipo',
            color='Cuenta',              # Nombre del cluster
            orientation='v',            
            title="Comparativo Ingresos vs Egresos",
            labels={'Monto': 'Monto', 'Tipo': 'Tipo', 'Cuenta':'Cuenta'})

    fig_2.update_xaxes(showticklabels=False)

     

    

#---------------------------------------------------------------------------------------------------------------------------    
    # Crear figura
    fig_3 = go.Figure()

    # Agregar la venta total
    total_data = dff_fig3.groupby('Fecha')['Monto'].sum().reset_index()
    fig_3.add_trace(go.Scatter(
        x=total_data['Fecha'],
        y=total_data['Monto'],
        mode='lines',
        name='Total',
        line=dict(width=3, color='red')  # Línea más gruesa y roja para diferenciarla
    ))

    # Agregar cada cluster con visibilidad oculta por defecto
    for Cuenta in dff_fig3['Cuenta'].unique():
        cluster_data = dff_fig3[dff_fig3['Cuenta'] == Cuenta]
        fig_3.add_trace(go.Scatter(
            x=cluster_data['Fecha'],
            y=cluster_data['Monto'],
            mode='lines',
            name=str(Cuenta),
            visible="legendonly"  # Hace que los clusters estén ocultos al inicio
        ))

    # Configuración del gráfico
    fig_3.update_layout(
        title="Evolucion de Egresos",
        xaxis_title='Fecha',
        yaxis_title='Egresos'
    )
 
#--------------------------------------------------------------------------------------------------------------------------------
#    
# Crear el gráfico Sunburst
    fig_4 = go.Figure(go.Sunburst(
        labels=labels,
        parents=parents,
        values=values,
        branchvalues="total"
    ))

    # Configurar el layout
    fig_4.update_layout(
        title="Participacion de Ingresos y Egresos",
        sunburstcolorway=["#636EFA", "#EF553B", "#00CC96"]
    )


    # Configurar el layout
    fig_4.update_layout(
        title="Participacion de Ingresos y Egresos",
        sunburstcolorway=["#636EFA", "#EF553B", "#00CC96", "#FFA15A", "#AB63FA"]
    )




    # Para la tabla se usan los datos filtrados
    data_table = df_filtrado.to_dict('records')
    columns=[
                {'name': 'Factura', 'id': 'Factura'},
                {'name': 'Fecha', 'id': 'Fecha'},
                {'name': 'Empresa', 'id': 'Empresa'},
                {'name': 'Tipo', 'id': 'Tipo'},
                {'name': 'Cuenta', 'id': 'Cuenta'},
                {'name': 'Monto', 'id': 'Monto'}
            ]

    # Retornar los outputs en el orden definido, incluyendo la vista previa de datos filtrados
    return fig_1, fig_2, fig_3, fig_4, data_table, columns
    

    

if __name__ == '__main__':
    app.run_server(port=8058)


# In[ ]:



