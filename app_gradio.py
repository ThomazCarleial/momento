import gradio as gr
import numpy as np
import plotly.graph_objects as go


def base_ortonormal(direcao):
    norm = np.linalg.norm(direcao)
    if norm == 0:
        return np.array([1, 0, 0]), np.array([0, 1, 0])
    n = direcao / norm
    auxiliar = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(auxiliar, n)) > 0.9:
        auxiliar = np.array([0.0, 1.0, 0.0])
    u = np.cross(n, auxiliar)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    v /= np.linalg.norm(v)
    return u, v


def gerar_campo_3d(fx, fy, fz, px, py, pz, raio, n_planos, n_aneis, preencher_disco):
    P = np.array([px, py, pz], dtype=float)
    F = np.array([fx, fy, fz], dtype=float)
    norm_F = np.linalg.norm(F)

    fig = go.Figure()

    if norm_F < 1e-6:
        fig.update_layout(title="A Força F não pode ser um vetor nulo (0, 0, 0).", height=650)
        return fig

    n = F / norm_F
    u, v = base_ortonormal(F)
    espacamento = 1.5
    ts = (np.arange(n_planos) - (n_planos - 1) / 2.0) * espacamento
    theta = np.linspace(0, 2 * np.pi, 100)

    # Pontos O equiespaçados de 60° (6 pontos por anel) em torno de Q.
    # Vários anéis concêntricos (raios diferentes) aumentam o nº de setas
    # mantendo o espaçamento angular de 60° entre pontos de um mesmo anel.
    angulos_O = np.arange(6) * (np.pi / 3)  # 0, 60, 120, 180, 240, 300 graus
    n_aneis = max(int(n_aneis), 1)
    raios_aneis = raio * (np.arange(1, n_aneis + 1) / n_aneis)

    # ---- Linha de ação de F (vermelha) ----
    t_a, t_b = ts.min() - espacamento, ts.max() + espacamento
    p_ini, p_fim = P + t_a * n, P + t_b * n
    fig.add_trace(go.Scatter3d(
        x=[p_ini[0], p_fim[0]], y=[p_ini[1], p_fim[1]], z=[p_ini[2], p_fim[2]],
        mode='lines', line=dict(color='red', width=6), name='Linha de Ação de F'
    ))

    # ---- Vetor força F aplicado em P (vermelho) ----
    escala_F = espacamento * 0.9 / max(norm_F, 1e-9)
    F_ponta = P + F * escala_F
    fig.add_trace(go.Scatter3d(
        x=[P[0], F_ponta[0]], y=[P[1], F_ponta[1]], z=[P[2], F_ponta[2]],
        mode='lines+markers', line=dict(color='red', width=8),
        marker=dict(size=[0, 4], color='red', symbol='diamond'),
        name='Vetor Força F'
    ))

    fig.add_trace(go.Scatter3d(
        x=[P[0]], y=[P[1]], z=[P[2]],
        mode='markers+text', marker=dict(size=8, color='red'),
        text=['P'], textposition='top center', name='Ponto P'
    ))

    # Coleta de dados: O, Q, ponta do vetor M(O), módulo de M
    dados = []
    for t in ts:
        Q = P + t * n
        for r_anel in raios_aneis:
            for ang in angulos_O:
                O = Q + r_anel * (np.cos(ang) * u + np.sin(ang) * v)
                M = np.cross(P - O, F)
                mod_M = np.linalg.norm(M)
                ponta = O + M * 0.15
                dados.append((O, Q, ponta, mod_M))

    # ---- Planos circulares (normais à linha de ação) ----
    for idx, t in enumerate(ts):
        Q = P + t * n
        cx = Q[0] + raio * (np.cos(theta) * u[0] + np.sin(theta) * v[0])
        cy = Q[1] + raio * (np.cos(theta) * u[1] + np.sin(theta) * v[1])
        cz = Q[2] + raio * (np.cos(theta) * u[2] + np.sin(theta) * v[2])

        fig.add_trace(go.Scatter3d(
            x=cx, y=cy, z=cz, mode='lines',
            line=dict(color='rgba(70,130,180,0.8)', width=3),
            name='Plano Normal a F' if idx == 0 else None,
            showlegend=(idx == 0)
        ))

        if preencher_disco:
            r_g = np.linspace(0, raio, 15)
            rm, tm = np.meshgrid(r_g, theta)
            fig.add_trace(go.Surface(
                x=Q[0] + rm * (np.cos(tm) * u[0] + np.sin(tm) * v[0]),
                y=Q[1] + rm * (np.cos(tm) * u[1] + np.sin(tm) * v[1]),
                z=Q[2] + rm * (np.cos(tm) * u[2] + np.sin(tm) * v[2]),
                colorscale=[[0, 'rgba(100,149,237,0.15)'], [1, 'rgba(100,149,237,0.15)']],
                showscale=False, hoverinfo='skip', showlegend=False
            ))

        # Ponto Q de cada plano
        fig.add_trace(go.Scatter3d(
            x=[Q[0]], y=[Q[1]], z=[Q[2]],
            mode='markers+text', marker=dict(size=5, color='dimgray'),
            text=['Q'], textposition='bottom center',
            showlegend=False,
            hovertemplate=f'Q (t={t:.2f})<extra></extra>'
        ))

    # ---- Retas tracejadas verdes: O -> Q ----
    xg, yg, zg = [], [], []
    for O, Q, ponta, mod_M in dados:
        xg += [O[0], Q[0], None]
        yg += [O[1], Q[1], None]
        zg += [O[2], Q[2], None]
    fig.add_trace(go.Scatter3d(
        x=xg, y=yg, z=zg, mode='lines',
        line=dict(color='green', width=3, dash='dash'),
        name='O → Q', showlegend=True
    ))

    # ---- Retas tracejadas azuis: M(O) -> Q ----
    xb, yb, zb = [], [], []
    for O, Q, ponta, mod_M in dados:
        xb += [ponta[0], Q[0], None]
        yb += [ponta[1], Q[1], None]
        zb += [ponta[2], Q[2], None]
    fig.add_trace(go.Scatter3d(
        x=xb, y=yb, z=zb, mode='lines',
        line=dict(color='blue', width=3, dash='dash'),
        name='M(O) → Q', showlegend=True
    ))

    # ---- Vetores M(O) em preto (de O até a ponta) ----
    xm, ym, zm = [], [], []
    for O, Q, ponta, mod_M in dados:
        xm += [O[0], ponta[0], None]
        ym += [O[1], ponta[1], None]
        zm += [O[2], ponta[2], None]
    fig.add_trace(go.Scatter3d(
        x=xm, y=ym, z=zm, mode='lines',
        line=dict(color='black', width=5),
        name='Vetores M(O)', showlegend=True
    ))

    # ---- Pontos O e pontas de M(O), com hover mostrando o módulo ----
    fig.add_trace(go.Scatter3d(
        x=[d[0][0] for d in dados], y=[d[0][1] for d in dados], z=[d[0][2] for d in dados],
        mode='markers', marker=dict(size=4, color='darkgreen'),
        name='Pontos O',
        customdata=[d[3] for d in dados],
        hovertemplate='O(x,y,z)<br>|M| = %{customdata:.3f} N·m<extra></extra>'
    ))
    fig.add_trace(go.Scatter3d(
        x=[d[2][0] for d in dados], y=[d[2][1] for d in dados], z=[d[2][2] for d in dados],
        mode='markers', marker=dict(size=4, color='black'),
        name='Pontos M(O)',
        customdata=[d[3] for d in dados],
        hovertemplate='M(O)<br>|M| = %{customdata:.3f} N·m<extra></extra>'
    ))

    fig.update_layout(
        scene=dict(aspectmode='data'),
        height=680,
        margin=dict(l=0, r=0, b=0, t=30),
        legend=dict(x=0.01, y=0.99)
    )
    return fig


with gr.Blocks(title="Campo de Momentos 3D com Discos") as demo:
    gr.Markdown("# Campo de Momentos 3D com Planos Circulares")
    gr.Markdown(
        "F e sua linha de ação em **vermelho**; vetores M(O) em **preto**; "
        "retas O→Q em **verde tracejado**; retas M(O)→Q em **azul tracejado**. "
        "Em cada plano normal a F, os pontos O formam anéis concêntricos de 6 pontos, "
        "sempre equiespaçados de **60°** dentro de cada anel."
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Vetor Força (F)")
            fx = gr.Number(value=1.0, label="Fx")
            fy = gr.Number(value=2.0, label="Fy")
            fz = gr.Number(value=3.0, label="Fz")

            gr.Markdown("### Ponto de Aplicação (P)")
            px = gr.Number(value=0.0, label="Px")
            py = gr.Number(value=0.0, label="Py")
            pz = gr.Number(value=0.0, label="Pz")

            gr.Markdown("### Geometria do Cilindro")
            raio = gr.Slider(minimum=0.5, maximum=3.0, value=1.2, label="Raio do Cilindro")
            n_planos = gr.Slider(minimum=1, maximum=9, value=5, step=1, label="Nº de Planos")
            n_aneis = gr.Slider(minimum=1, maximum=5, value=2, step=1,
                                 label="Nº de Anéis por Plano (6 setas cada, a 60°)")
            preencher = gr.Checkbox(value=True, label="Preencher superfície dos discos")

            btn = gr.Button("Atualizar Campo", variant="primary")

        with gr.Column(scale=3):
            plot = gr.Plot(label="Visualização 3D")

    inputs = [fx, fy, fz, px, py, pz, raio, n_planos, n_aneis, preencher]
    btn.click(fn=gerar_campo_3d, inputs=inputs, outputs=plot)
    demo.load(fn=gerar_campo_3d, inputs=inputs, outputs=plot)

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
