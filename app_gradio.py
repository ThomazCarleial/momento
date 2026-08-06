import gradio as gr
import numpy as np
import plotly.graph_objects as go
import plotly.colors as pc

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

def gerar_campo_3d(fx, fy, fz, px, py, pz, raio, n_planos, n_pontos, preencher_disco):
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
    n_pts = max(int(n_pontos), 1)
    idx_pts = np.arange(n_pts)
    golden_angle = np.pi * (3 - np.sqrt(5))
    raios_pontos = raio * np.sqrt((idx_pts + 0.5) / n_pts)
    angulos_pontos = idx_pts * golden_angle

    dados = []
    for t in ts:
        Q = P + t * n
        for r_pt, ang in zip(raios_pontos, angulos_pontos):
            O = Q + r_pt * (np.cos(ang) * u + np.sin(ang) * v)
            M = np.cross(P - O, F)
            mod_M = np.linalg.norm(M)
            dados.append((O, O + M * 0.15, mod_M))

    todos_mod = np.array([d[2] for d in dados])
    M_min, M_max = todos_mod.min(), todos_mod.max()
    escala_ref = max(abs(M_max), abs(M_min), 1e-9)
    variacao_real = (M_max - M_min) > escala_ref * 1e-6
    M_range = (M_max - M_min) if variacao_real else 1.0
    t_a, t_b = ts.min() - espacamento, ts.max() + espacamento
    p_ini, p_fim = P + t_a * n, P + t_b * n
    fig.add_trace(go.Scatter3d(
        x=[p_ini[0], p_fim[0]], y=[p_ini[1], p_fim[1]], z=[p_ini[2], p_fim[2]],
        mode='lines', line=dict(color='crimson', width=6), name='Linha de Ação / Força F'
    ))

    fig.add_trace(go.Scatter3d(
        x=[P[0]], y=[P[1]], z=[P[2]],
        mode='markers+text', marker=dict(size=8, color='crimson'),
        text=['P'], textposition='top center', name='Ponto P'
    ))
    for idx, t in enumerate(ts):
        Q = P + t * n
        cx = Q[0] + raio * (np.cos(theta) * u[0] + np.sin(theta) * v[0])
        cy = Q[1] + raio * (np.cos(theta) * u[1] + np.sin(theta) * v[1])
        cz = Q[2] + raio * (np.cos(theta) * u[2] + np.sin(theta) * v[2])

        fig.add_trace(go.Scatter3d(
            x=cx, y=cy, z=cz, mode='lines',
            line=dict(color='rgba(70,130,180,0.8)', width=3),
            name='Plano Circular' if idx == 0 else None,
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

    if variacao_real:
        N_BUCKETS = 20
        cores_buckets = pc.sample_colorscale('plasma', [i / max(N_BUCKETS - 1, 1) for i in range(N_BUCKETS)])

        buckets = {i: {'x': [], 'y': [], 'z': [], 'customdata': []} for i in range(N_BUCKETS)}

        for O, ponta, mod_M in dados:
            t_norm = (mod_M - M_min) / M_range
            bi = int(np.clip(t_norm * (N_BUCKETS - 1), 0, N_BUCKETS - 1))
            buckets[bi]['x'] += [O[0], ponta[0], None]
            buckets[bi]['y'] += [O[1], ponta[1], None]
            buckets[bi]['z'] += [O[2], ponta[2], None]
            buckets[bi]['customdata'] += [mod_M, mod_M, None]

        for i, b in buckets.items():
            if not b['x']:
                continue
            fig.add_trace(go.Scatter3d(
                x=b['x'], y=b['y'], z=b['z'],
                mode='lines', line=dict(color=cores_buckets[i], width=5),
                showlegend=False,
                customdata=b['customdata'],
                hovertemplate='|M| = %{customdata:.3f}<extra></extra>'
            ))

        marker_kwargs = dict(
            size=5,
            color=[d[2] for d in dados],
            colorscale='plasma',
            cmin=M_min, cmax=M_max,
            showscale=True,
            colorbar=dict(
                title=dict(text='|M(O)| (N·m)', side='right', font=dict(size=13)),
                thickness=18, len=0.75, x=1.02,
                tickfont=dict(size=11), outlinewidth=1,
            )
        )
        hover_pontos = '|M| = %{marker.color:.3f}<extra></extra>'
    else:
        mod_constante = float(todos_mod.mean())
        cor_unica = pc.sample_colorscale('plasma', [0.55])[0]

        xs, ys, zs = [], [], []
        for O, ponta, _ in dados:
            xs += [O[0], ponta[0], None]
            ys += [O[1], ponta[1], None]
            zs += [O[2], ponta[2], None]

        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode='lines', line=dict(color=cor_unica, width=5),
            showlegend=False,
            hovertemplate=f'|M| = {mod_constante:.3f} (constante)<extra></extra>'
        ))

        marker_kwargs = dict(size=5, color=cor_unica, showscale=False)
        hover_pontos = f'|M| = {mod_constante:.3f} (constante)<extra></extra>'

    fig.add_trace(go.Scatter3d(
        x=[d[0][0] for d in dados],
        y=[d[0][1] for d in dados],
        z=[d[0][2] for d in dados],
        mode='markers',
        marker=marker_kwargs,
        showlegend=False,
        hovertemplate=hover_pontos
    ))

    if not variacao_real:
        fig.add_annotation(
            text=f"|M| constante = raio × |F| = {mod_constante:.3f} N·m (teorema de Varignon: "
                 f"todos os pontos O estão à mesma distância perpendicular à linha de ação)",
            xref="paper", yref="paper", x=0.5, y=0.02, showarrow=False,
            font=dict(size=11, color="gray")
        )

    fig.update_layout(
        scene=dict(aspectmode='data'),
        height=680,
        margin=dict(l=0, r=0, b=0, t=30),
        legend=dict(x=0.01, y=0.99)
    )
    return fig


with gr.Blocks(title="Campo de Momentos 3D com Discos") as demo:
    gr.Markdown("# Campo de Momentos 3D com Planos Circulares")

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
            raio   = gr.Slider(minimum=0.5, maximum=3.0, value=1.2, label="Raio do Cilindro")
            n_planos = gr.Slider(minimum=1, maximum=9, value=5, step=1, label="Nº de Planos")
            n_pontos = gr.Slider(minimum=3, maximum=12, value=6, step=1, label="Pontos por Plano")
            preencher = gr.Checkbox(value=True, label="Preencher superfície dos discos")

            btn = gr.Button("Atualizar Campo", variant="primary")

        with gr.Column(scale=3):
            plot = gr.Plot(label="Visualização 3D")

    inputs = [fx, fy, fz, px, py, pz, raio, n_planos, n_pontos, preencher]
    btn.click(fn=gerar_campo_3d, inputs=inputs, outputs=plot)
    demo.load(fn=gerar_campo_3d, inputs=inputs, outputs=plot)

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
