# Databricks notebook source
# DBTITLE 1,Ingestão Bronze - Futebol Europeu (4 Épocas)
import pandas as pd
from pyspark.sql import functions as F

# 1. Mapeamento das ligas e épocas (4 épocas: 2020/21 a 2023/24)
leagues = {
    "primeira_liga_pt": "P1",
    "la_liga_es": "SP1",
    "premier_league_uk": "E0",
    "serie_a_it": "I1",
    "bundesliga_de": "D1"
}

seasons = {
    "2020/21": "2021",
    "2021/22": "2122",
    "2022/23": "2223",
    "2023/24": "2324"
}

print("🚀 A iniciar a ingestão de dados desportivos europeus (4 épocas)...")

dataframes_list = []

# 2. Download e Processamento: iterar sobre épocas e ligas
for season_name, season_code in seasons.items():
    for league_name, league_code in leagues.items():
        url = f"https://www.football-data.co.uk/mmz4281/{season_code}/{league_code}.csv"
        try:
            # Download e leitura do CSV com pandas
            pdf = pd.read_csv(url)
            df_raw = spark.createDataFrame(pdf)
        except Exception as e:
            print(f"❌ Erro: {season_name} / {league_name}: {str(e)[:80]}")
            continue
        
        # Manter todas as colunas + adicionar league_name e season
        df_clean = df_raw.select("*", 
            F.lit(league_name).alias("league_name"),
            F.lit(season_name).alias("season"))
        
        try:
            # Ação (eager) - count dispara as transformações lazy
            count = df_clean.count()
            dataframes_list.append(df_clean)
            print(f"✅ {season_name} / {league_name}: {count} jogos")
        except Exception as e:
            print(f"❌ Erro: {season_name} / {league_name}: {str(e)[:80]}")

# 3. Unificação de todas as épocas e ligas numa única tabela Delta
df_europe_all = dataframes_list[0]
for df in dataframes_list[1:]:
    df_europe_all = df_europe_all.unionByName(df, allowMissingColumns=True)

df_europe_all = df_europe_all.withColumn("_ingestion_timestamp", F.current_timestamp())

df_europe_all.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable("bronze_european_football_master")
total = df_europe_all.count()
print(f"\n🎉 SUCESSO! Tabela 'bronze_european_football_master' criada com {total} jogos de {len(seasons)} épocas.")

# COMMAND ----------

# DBTITLE 1,Pergunta 1 - Vantagem de jogar em casa
from pyspark.sql import functions as F
from pyspark.sql.window import Window

df = spark.table("bronze_european_football_master")

# --- PERGUNTA 1: VANTAGEM DE JOGAR EM CASA POR LIGA ---

# Pergunta: Em média, quantos golos marcam as equipas da casa vs. visitantes, e a percentagem de vitórias caseiras varia entre as 5 ligas europeias?

# Objetivo da Análise: Quantificar a vantagem de jogar em casa em cada liga (golos marcados, % vitórias, diferença casa-fora) para identificar quais ligas têm maior 'home advantage' e perceber se este fator deve ser incluído como feature em modelos preditivos.
# Métricas: avg_golos_casa, avg_golos_fora, pct_vitoria_casa, diferenca_golos

df_home_adv = (df.groupBy("league_name")
    .agg(
        F.round(F.avg("FTHG"), 2).alias("avg_golos_casa"),
        F.round(F.avg("FTAG"), 2).alias("avg_golos_fora"),
        F.count(F.when(F.col("FTR") == "H", True)).alias("vitorias_casa"),
        F.count(F.when(F.col("FTR") == "D", True)).alias("empates"),
        F.count(F.when(F.col("FTR") == "A", True)).alias("vitorias_fora"),
        F.count("*").alias("total_jogos")
    )
    .withColumn("pct_vitoria_casa", F.round(F.col("vitorias_casa") / F.col("total_jogos") * 100, 1))
    .withColumn("diferenca_golos", F.round(F.col("avg_golos_casa") - F.col("avg_golos_fora"), 2))
    .orderBy(F.desc("pct_vitoria_casa"))
)

display(df_home_adv)

# COMMAND ----------

# DBTITLE 1,Pergunta 2 - Equipas com melhor aproveitamento em casa
# --- PERGUNTA 2: EQUIPAS COM MELHOR APROVEITAMENTO EM CASA ---

# Pergunta: Que percentagem de pontos cada equipa conquistou jogando em casa vs. fora? Existem equipas claramente dependentes do público caseiro?

# Objetivo da Análise: Mapear o desempenho de cada equipa em casa vs. fora (pontos conquistados sobre o máximo possível) para identificar equipas fortes em casa, equipas equilibradas e equipas dependentes do fator casa — informação crucial para prever resultados com base no local do jogo.

# Métricas: pct_aproveitamento_casa, pct_aproveitamento_fora, diff_aproveitamento (Vitória=3pts, Empate=1pt, Derrota=0pts)

# Pontos em casa (por HomeTeam)
df_home_points = (df.groupBy("league_name", "HomeTeam")
    .agg(
        F.sum(F.when(F.col("FTR") == "H", 3).when(F.col("FTR") == "D", 1).otherwise(0)).alias("pontos_casa"),
        F.count("*").alias("jogos_casa")
    )
    .withColumn("pct_aproveitamento_casa", F.round(F.col("pontos_casa") / (F.col("jogos_casa") * 3) * 100, 1))
)

# Pontos fora (por AwayTeam)
df_away_points = (df.groupBy("league_name", "AwayTeam")
    .agg(
        F.sum(F.when(F.col("FTR") == "A", 3).when(F.col("FTR") == "D", 1).otherwise(0)).alias("pontos_fora"),
        F.count("*").alias("jogos_fora")
    )
    .withColumn("pct_aproveitamento_fora", F.round(F.col("pontos_fora") / (F.col("jogos_fora") * 3) * 100, 1))
)

# Join casa + fora
df_team_perf = (df_home_points.join(
        df_away_points,
        (df_home_points.HomeTeam == df_away_points.AwayTeam) & (df_home_points.league_name == df_away_points.league_name),
        "inner"
    )
    .select(
        df_home_points.league_name,
        df_home_points.HomeTeam.alias("equipa"),
        "pontos_casa", "jogos_casa", "pct_aproveitamento_casa",
        "pontos_fora", "jogos_fora", "pct_aproveitamento_fora"
    )
    .withColumn("diff_aproveitamento", F.round(F.col("pct_aproveitamento_casa") - F.col("pct_aproveitamento_fora"), 1))
    .orderBy(F.desc("pct_aproveitamento_casa"))
)

display(df_team_perf.limit(20))

# COMMAND ----------

# DBTITLE 1,Pergunta 3 - Shots on target determinam resultados
# --- PERGUNTA 3: REMATES À BALIZA DETERMINAM RESULTADOS? ---
# Pergunta: Qual a correlação entre remates à baliza (HST/AST) e golos marcados (FTHG/FTAG)? Equipas que rematam mais à baliza ganham mais vezes?
# Objetivo da Análise: Avaliar se os remates à baliza são um bom preditor de golos e resultados, comparando as médias de HST/AST por resultado final (H/D/A) e calculando correlações. Determinar se esta estatística deve ser usada como feature em modelos de previsão de vitórias.
# Métricas: avg_remates_casa, avg_remates_fora, correlação HST-FTHG (0.57), correlação AST-FTAG (0.59)

df_shots = (df.groupBy("FTR")
    .agg(
        F.round(F.avg("HST"), 2).alias("avg_remates_casa"),
        F.round(F.avg("AST"), 2).alias("avg_remates_fora"),
        F.round(F.avg("FTHG"), 2).alias("avg_golos_casa"),
        F.round(F.avg("FTAG"), 2).alias("avg_golos_fora"),
        F.count("*").alias("n_jogos")
    )
    .orderBy("FTR")
)

print("=== Médias por resultado final (FTR) ===")
print("H = Vitória casa | D = Empate | A = Vitória fora\n")
display(df_shots)

# Correlação entre remates à baliza e golos marcados
print("\n=== Correlação: remates à baliza vs golos marcados ===")
df_corr = df.select(
    F.corr("HST", "FTHG").alias("corr_remates_casa_golos_casa"),
    F.corr("AST", "FTAG").alias("corr_remates_fora_golos_fora"),
    F.corr("HS", "FTHG").alias("corr_remates_totais_casa_golos"),
    F.corr("AS", "FTAG").alias("corr_remates_totais_fora_golos")
)
display(df_corr)

# COMMAND ----------

# DBTITLE 1,Pergunta 4 - Odds dos bookies prevêem resultados
# --- PERGUNTA 4: AS ODDS DOS BOOKIES PREVEEM CORRETAMENTE OS RESULTADOS? ---
# Pergunta: Quando o favorito (menor odd) era a equipa da casa, quantas vezes ganhou realmente? A odd média (AvgH/AvgD/AvgA) é um bom preditor do resultado final (FTR)?
# Objetivo da Análise: Avaliar a capacidade preditiva das odds das casas de apostas comparando o favorito (menor odd) com o resultado real. Determinar a taxa de acerto por liga e se as odds podem ser usadas como feature num modelo de previsão de vitórias.
# Métricas: pct_acerto_favorito, pct_acerto_por_liga, odds_avg (AvgH/AvgD/AvgA)

# Identificar o favorito (menor odd) e comparar com o resultado real
df_odds = (df
    .filter(F.col("AvgH").isNotNull() & F.col("AvgD").isNotNull() & F.col("AvgA").isNotNull())
    .withColumn("favorito",
        F.when(F.col("AvgH") <= F.col("AvgA"), F.lit("H"))
         .otherwise(F.lit("A"))
    )
    .withColumn("acerto", F.when(F.col("favorito") == F.col("FTR"), 1).otherwise(0))
)

# Taxa de acerto global
df_odds_global = df_odds.agg(
    F.round(F.avg("acerto") * 100, 1).alias("pct_acerto_global"),
    F.count("*").alias("total_jogos_com_odds")
)
print("=== Taxa de acerto global do favorito (menor odd) ===")
display(df_odds_global)

# Taxa de acerto por liga
df_odds_league = (df_odds.groupBy("league_name")
    .agg(
        F.round(F.avg("acerto") * 100, 1).alias("pct_acerto"),
        F.count("*").alias("n_jogos")
    )
    .orderBy(F.desc("pct_acerto"))
)
print("\n=== Taxa de acerto do favorito por liga ===")
display(df_odds_league)

# COMMAND ----------

# DBTITLE 1,Pergunta 5 - Cartões e faltas influenciam resultados
# --- PERGUNTA 5: CARTÕES E FALTAS INFLUENCIAM O DESEMPENHO? ---

# Pergunta: Equipas que cometem mais faltas (HF/AF) ou recebem mais cartões vermelhos (HR/AR) perdem mais jogos? Qual a liga mais 'indisciplinada'?

# Objetivo da Análise: Medir o impacto de faltas e cartões nos resultados, identificando se equipas indisciplinadas perdem mais jogos. Comparar a indisciplina média entre ligas para perceber se este fator deve ser incluído como feature no modelo preditivo.

# Métricas: avg_faltas_casa (HF), avg_faltas_fora (AF), avg_cartoes_vermelhos_casa (HR), avg_cartoes_vermelhos_fora (AR), pct_derrota_com_vermelho

# Médias de faltas e cartões por resultado final
df_cards = (df.groupBy("FTR")
    .agg(
        F.round(F.avg("HF"), 2).alias("avg_faltas_casa"),
        F.round(F.avg("AF"), 2).alias("avg_faltas_fora"),
        F.round(F.avg("HY") + F.avg("AY"), 2).alias("avg_amarelos_total"),
        F.round(F.avg("HR"), 3).alias("avg_vermelhos_casa"),
        F.round(F.avg("AR"), 3).alias("avg_vermelhos_fora"),
        F.count("*").alias("n_jogos")
    )
    .orderBy("FTR")
)
print("=== Médias de faltas e cartões por resultado (FTR) ===")
print("H = Vitória casa | D = Empate | A = Vitória fora\n")
display(df_cards)

# Liga mais indisciplinada (média de faltas e cartões por jogo)
df_discipline = (df.groupBy("league_name")
    .agg(
        F.round(F.avg("HF") + F.avg("AF"), 2).alias("avg_faltas_jogo"),
        F.round(F.avg("HY") + F.avg("AY"), 2).alias("avg_amarelos_jogo"),
        F.round(F.avg("HR") + F.avg("AR"), 3).alias("avg_vermelhos_jogo")
    )
    .orderBy(F.desc("avg_faltas_jogo"))
)
print("\n=== Indisciplina média por liga (faltas + cartões por jogo) ===")
display(df_discipline)

# COMMAND ----------

# DBTITLE 1,Pergunta 6 - Over 2.5 golos
# --- PERGUNTA 6: OVER 2.5 GOLOS — QUANDO É MAIS PROVÁVEL? ---
# Pergunta: Que combinações de estatísticas (remates, cantos, faltas) estão associadas a jogos com mais de 2.5 golos? A Bundesliga tem mais jogos de 'over' do que a Serie A?
# Objetivo da Análise: Identificar fatores estatísticos associados a jogos de muitos golos (>2.5), comparar a % de over 2.5 entre ligas e perceber que variáveis podem ser features num modelo de previsão de total de golos.
# Métricas: pct_over25_por_liga, avg_remates_over, avg_cantos_over, avg_faltas_over

# Classificar jogos: Over 2.5 golos (FTHG + FTAG > 2)
df_over = (df
    .withColumn("over_25", F.when(F.col("FTHG") + F.col("FTAG") > 2, 1).otherwise(0))
    .withColumn("total_golos", F.col("FTHG") + F.col("FTAG"))
)

# % Over 2.5 por liga
df_over_league = (df_over.groupBy("league_name")
    .agg(
        F.round(F.avg("over_25") * 100, 1).alias("pct_over_25"),
        F.round(F.avg("total_golos"), 2).alias("avg_golos_jogo"),
        F.count("*").alias("n_jogos")
    )
    .orderBy(F.desc("pct_over_25"))
)
print("=== % Over 2.5 golos por liga ===")
display(df_over_league)

# Estatísticas médias: Over vs Under 2.5
df_over_stats = (df_over.groupBy("over_25")
    .agg(
        F.round(F.avg("HS") + F.avg("AS"), 2).alias("avg_remates_totais"),
        F.round(F.avg("HST") + F.avg("AST"), 2).alias("avg_remates_baliza"),
        F.round(F.avg("HC") + F.avg("AC"), 2).alias("avg_cantos"),
        F.round(F.avg("HF") + F.avg("AF"), 2).alias("avg_faltas"),
        F.count("*").alias("n_jogos")
    )
    .orderBy("over_25")
)
print("\n=== Estatísticas médias: Over (1) vs Under (0) 2.5 golos ===")
display(df_over_stats)

# COMMAND ----------

# DBTITLE 1,Pergunta 7 - Padrões entre ligas
# --- PERGUNTA 7: PADRÕES QUE DISTINGUEM AS LIGAS ENTRE SI ---

# Pergunta: Em que liga se marcam mais golos por jogo? Onde há mais empates? Qual a liga com odds mais imprevisíveis (maior variância entre favorito e vencedor real)?

# Objetivo da Análise: Comparar o perfil de jogo das 5 ligas europeias (golos, empates, imprevisibilidade das odds) para identificar padrões distintos que possam justificar modelos separados por liga ou features de liga no modelo agregado.

# Métricas: avg_golos_jogo, pct_empates, pct_acerto_favorito, avg_odd_diferencial

# Resumo comparativo por liga
df_league_profile = (df
    .filter(F.col("AvgH").isNotNull() & F.col("AvgD").isNotNull() & F.col("AvgA").isNotNull())
    .withColumn("total_golos", F.col("FTHG") + F.col("FTAG"))
    .withColumn("favorito",
        F.when(F.col("AvgH") <= F.col("AvgA"), F.lit("H"))
         .otherwise(F.lit("A"))
    )
    .withColumn("acerto", F.when(F.col("favorito") == F.col("FTR"), 1).otherwise(0))
    .groupBy("league_name")
    .agg(
        F.round(F.avg("total_golos"), 2).alias("avg_golos_jogo"),
        F.round(F.avg(F.when(F.col("FTR") == "D", 1).otherwise(0)) * 100, 1).alias("pct_empates"),
        F.round(F.avg("acerto") * 100, 1).alias("pct_acerto_favorito"),
        F.round(F.avg("AvgD"), 2).alias("avg_odd_empate"),
        F.count("*").alias("n_jogos")
    )
    .orderBy(F.desc("avg_golos_jogo"))
)
print("=== Perfil comparativo das 5 ligas europeias ===")
display(df_league_profile)

# COMMAND ----------

# DBTITLE 1,Pergunta 8 - Equipas da Liga Portuguesa
# --- PERGUNTA 8: QUAIS SÃO TODAS AS EQUIPAS DA LIGA PORTUGUESA? ---

# Pergunta: Quantas equipas diferentes participaram na Primeira Liga portuguesa nas 4 épocas analisadas? Quantos jogos disputou cada uma e em quantas épocas esteve presente?

# Objetivo da Análise: Listar todas as equipas que apareceram na Primeira Liga portuguesa (primeira_liga_pt) ao longo das 4 épocas (2020/21 a 2023/24), calculando o total de jogos, épocas disputadas e golos marcados para ter uma visão geral do panorama competitivo da liga.

# Métricas: total_equipas, jogos_totais, epocas_disputadas, golos_marcados

df_pt = spark.table("bronze_european_football_master").filter(F.col("league_name") == "primeira_liga_pt")

# Equipas em casa
df_home_teams = (df_pt.groupBy("HomeTeam")
    .agg(
        F.count("*").alias("jogos_casa"),
        F.sum("FTHG").alias("golos_casa"),
        F.collect_set("season").alias("epocas_casa")
    )
)

# Equipas fora
df_away_teams = (df_pt.groupBy("AwayTeam")
    .agg(
        F.count("*").alias("jogos_fora"),
        F.sum("FTAG").alias("golos_fora"),
        F.collect_set("season").alias("epocas_fora")
    )
)

# Join casa + fora para ter visão completa por equipa
df_pt_teams = (df_home_teams.join(df_away_teams,
        df_home_teams.HomeTeam == df_away_teams.AwayTeam, "inner")
    .select(
        F.col("HomeTeam").alias("equipa"),
        (F.col("jogos_casa") + F.col("jogos_fora")).alias("jogos_totais"),
        (F.col("golos_casa") + F.col("golos_fora")).alias("golos_marcados"),
        F.size(F.array_distinct(F.array_union(F.col("epocas_casa"), F.col("epocas_fora")))).alias("n_epocas")
    )
    .orderBy(F.desc("golos_marcados"))
)

print("=== Equipas da Liga Portuguesa (Primeira Liga) — 4 Épocas ===")
print(f"Total de equipas: {df_pt_teams.count()}\n")
display(df_pt_teams)

# COMMAND ----------

# DBTITLE 1,Pergunta 9 - Ranking desempenho Liga Portuguesa
# --- PERGUNTA 9: RANKING DE DESEMPENHO DAS EQUIPAS DA LIGA PORTUGUESA ---

# Pergunta: Qual a equipa com mais vitórias, empates e derrotas na Liga Portuguesa nas 4 épocas? Qual marcou mais golos, qual sofreu menos e qual tem o melhor aproveitamento de pontos?

# Objetivo da Análise: Construir um ranking completo de desempenho das equipas da Primeira Liga portuguesa, calculando vitórias, empates, derrotas (em casa e fora), golos marcados, golos sofridos, diferença de golos e pontos conquistados, para identificar as equipas mais dominantes e as mais vulneráveis ao longo das 4 épocas.

# Métricas: vitorias, empates, derrotas, golos_marcados, golos_sofridos, diferenca_golos, pontos, pct_aproveitamento

# Pontos e resultados por equipa em casa
df_pt_home = (df_pt.groupBy("HomeTeam")
    .agg(
        F.sum(F.when(F.col("FTR") == "H", 1).otherwise(0)).alias("vitorias_casa"),
        F.sum(F.when(F.col("FTR") == "D", 1).otherwise(0)).alias("empates_casa"),
        F.sum(F.when(F.col("FTR") == "A", 1).otherwise(0)).alias("derrotas_casa"),
        F.sum("FTHG").alias("golos_marcados_casa"),
        F.sum("FTAG").alias("golos_sofridos_casa"),
        F.sum(F.when(F.col("FTR") == "H", 3).when(F.col("FTR") == "D", 1).otherwise(0)).alias("pontos_casa"),
        F.count("*").alias("jogos_casa")
    )
)

# Pontos e resultados por equipa fora
df_pt_away = (df_pt.groupBy("AwayTeam")
    .agg(
        F.sum(F.when(F.col("FTR") == "A", 1).otherwise(0)).alias("vitorias_fora"),
        F.sum(F.when(F.col("FTR") == "D", 1).otherwise(0)).alias("empates_fora"),
        F.sum(F.when(F.col("FTR") == "H", 1).otherwise(0)).alias("derrotas_fora"),
        F.sum("FTAG").alias("golos_marcados_fora"),
        F.sum("FTHG").alias("golos_sofridos_fora"),
        F.sum(F.when(F.col("FTR") == "A", 3).when(F.col("FTR") == "D", 1).otherwise(0)).alias("pontos_fora"),
        F.count("*").alias("jogos_fora")
    )
)

# Join e cálculo de métricas totais
df_pt_ranking = (df_pt_home.join(df_pt_away,
        df_pt_home.HomeTeam == df_pt_away.AwayTeam, "inner")
    .select(
        F.col("HomeTeam").alias("equipa"),
        (F.col("vitorias_casa") + F.col("vitorias_fora")).alias("vitorias"),
        (F.col("empates_casa") + F.col("empates_fora")).alias("empates"),
        (F.col("derrotas_casa") + F.col("derrotas_fora")).alias("derrotas"),
        (F.col("golos_marcados_casa") + F.col("golos_marcados_fora")).alias("golos_marcados"),
        (F.col("golos_sofridos_casa") + F.col("golos_sofridos_fora")).alias("golos_sofridos"),
        (F.col("pontos_casa") + F.col("pontos_fora")).alias("pontos"),
        (F.col("jogos_casa") + F.col("jogos_fora")).alias("jogos_totais")
    )
    .withColumn("diferenca_golos", F.col("golos_marcados") - F.col("golos_sofridos"))
    .withColumn("pct_aproveitamento", F.round(F.col("pontos") / (F.col("jogos_totais") * 3) * 100, 1))
    .orderBy(F.desc("pontos"))
)

print("=== Ranking de Desempenho — Liga Portuguesa (4 Épocas) ===")
print("Ordenado por pontos conquistados (Vitória=3pts, Empate=1pt, Derrota=0pts)\n")
display(df_pt_ranking)

# COMMAND ----------

# DBTITLE 1,Pergunta 10 - Campeões e mais títulos da Liga Portuguesa
# --- PERGUNTA 10: QUEM TEM MAIS TÍTULOS DO CAMPEONATO? ---

# Pergunta: Quantas vezes cada equipa foi campeã da Primeira Liga nas 4 épocas analisadas? Qual a equipa com mais títulos?

# Objetivo da Análise: Determinar o campeão de cada época da Primeira Liga portuguesa (2020/21 a 2023/24) com base nos pontos conquistados (Vitória=3pts, Empate=1pt, Derrota=0pts), usando a diferença de golos como desempate. Contar o total de títulos por equipa para identificar a equipa mais vitoriosa do período.

# Métricas: epoca, campeao, pontos, diferenca_golos, total_titulos

from pyspark.sql.window import Window

# Pontos por equipa por época (casa)
df_pt_home_season = (df_pt.groupBy("season", "HomeTeam")
    .agg(
        F.sum(F.when(F.col("FTR") == "H", 3).when(F.col("FTR") == "D", 1).otherwise(0)).alias("pontos_casa"),
        F.sum("FTHG").alias("golos_marcados_casa"),
        F.sum("FTAG").alias("golos_sofridos_casa")
    )
)

# Pontos por equipa por época (fora)
df_pt_away_season = (df_pt.groupBy("season", "AwayTeam")
    .agg(
        F.sum(F.when(F.col("FTR") == "A", 3).when(F.col("FTR") == "D", 1).otherwise(0)).alias("pontos_fora"),
        F.sum("FTAG").alias("golos_marcados_fora"),
        F.sum("FTHG").alias("golos_sofridos_fora")
    )
)

# Join casa + fora: pontos totais e diferença de golos por época
df_pt_season = (df_pt_home_season.join(df_pt_away_season,
        (df_pt_home_season.season == df_pt_away_season.season) &
        (df_pt_home_season.HomeTeam == df_pt_away_season.AwayTeam), "inner")
    .select(
        df_pt_home_season.season.alias("epoca"),
        df_pt_home_season.HomeTeam.alias("equipa"),
        (F.col("pontos_casa") + F.col("pontos_fora")).alias("pontos"),
        ((F.col("golos_marcados_casa") + F.col("golos_marcados_fora")) -
         (F.col("golos_sofridos_casa") + F.col("golos_sofridos_fora"))).alias("dif_golos")
    )
)

# Campeão de cada época (mais pontos, desempate por diferença de golos)
window_spec = Window.partitionBy("epoca").orderBy(F.desc("pontos"), F.desc("dif_golos"))
df_champions = (df_pt_season
    .withColumn("rank", F.row_number().over(window_spec))
    .filter(F.col("rank") == 1)
    .drop("rank")
    .orderBy("epoca")
)

print("=== Campeões da Primeira Liga por Época ===")
display(df_champions)

# Contar títulos por equipa
df_titles = (df_champions.groupBy("equipa")
    .agg(
        F.count("*").alias("total_titulos"),
        F.collect_list("epoca").alias("epocas_vencidas")
    )
    .orderBy(F.desc("total_titulos"))
)

print("\n=== Ranking de Títulos da Primeira Liga (4 Épocas) ===")
display(df_titles)

# COMMAND ----------

# DBTITLE 1,Pergunta 11 - Campeões e mais títulos da La Liga Espanhola
# --- PERGUNTA 11: QUEM TEM MAIS TÍTULOS DA LA LIGA ESPANHOLA? ---

# Pergunta: Quantas vezes cada equipa foi campeã da Primeira Liga Espanhola nas 4 épocas analisadas? Qual a equipa com mais títulos?

# Objetivo da Análise: Determinar o campeão de cada época da Primeira Liga Espanhola (2020/21 a 2023/24) com base nos pontos conquistados (Vitória=3pts, Empate=1pt, Derrota=0pts), usando a diferença de golos como desempate. Contar o total de títulos por equipa para identificar a equipa mais vitoriosa do período.

# Métricas: epoca, campeao, pontos, diferenca_golos, total_titulos

from pyspark.sql import functions as F
from pyspark.sql.window import Window

df_es = spark.table("bronze_european_football_master").filter(F.col("league_name") == "la_liga_es")

# Pontos por equipa por época (casa)
df_es_home_season = (df_es.groupBy("season", "HomeTeam")
    .agg(
        F.sum(F.when(F.col("FTR") == "H", 3).when(F.col("FTR") == "D", 1).otherwise(0)).alias("pontos_casa"),
        F.sum("FTHG").alias("golos_marcados_casa"),
        F.sum("FTAG").alias("golos_sofridos_casa")
    )
)

# Pontos por equipa por época (fora)
df_es_away_season = (df_es.groupBy("season", "AwayTeam")
    .agg(
        F.sum(F.when(F.col("FTR") == "A", 3).when(F.col("FTR") == "D", 1).otherwise(0)).alias("pontos_fora"),
        F.sum("FTAG").alias("golos_marcados_fora"),
        F.sum("FTHG").alias("golos_sofridos_fora")
    )
)

# Join casa + fora: pontos totais e diferença de golos por época
df_es_season = (df_es_home_season.join(df_es_away_season,
        (df_es_home_season.season == df_es_away_season.season) &
        (df_es_home_season.HomeTeam == df_es_away_season.AwayTeam), "inner")
    .select(
        df_es_home_season.season.alias("epoca"),
        df_es_home_season.HomeTeam.alias("equipa"),
        (F.col("pontos_casa") + F.col("pontos_fora")).alias("pontos"),
        ((F.col("golos_marcados_casa") + F.col("golos_marcados_fora")) -
         (F.col("golos_sofridos_casa") + F.col("golos_sofridos_fora"))).alias("dif_golos")
    )
)

# Campeão de cada época (mais pontos, desempate por diferença de golos)
window_spec = Window.partitionBy("epoca").orderBy(F.desc("pontos"), F.desc("dif_golos"))
df_champions_es = (df_es_season
    .withColumn("rank", F.row_number().over(window_spec))
    .filter(F.col("rank") == 1)
    .drop("rank")
    .orderBy("epoca")
)

print("=== Campeões da La Liga Espanhola por Época ===")
display(df_champions_es)

# Contar títulos por equipa
df_titles_es = (df_champions_es.groupBy("equipa")
    .agg(
        F.count("*").alias("total_titulos"),
        F.collect_list("epoca").alias("epocas_vencidas")
    )
    .orderBy(F.desc("total_titulos"))
)

print("\n=== Ranking de Títulos da La Liga Espanhola (4 Épocas) ===")
display(df_titles_es)

# COMMAND ----------

# DBTITLE 1,Pergunta 12 - Ranking desempenho La Liga Espanhola
# --- PERGUNTA 12: RANKING DE DESEMPENHO DAS EQUIPAS DA LA LIGA ESPANHOLA ---

# Pergunta: Qual a equipa com mais vitórias, empates e derrotas na La Liga Espanhola nas 4 épocas? Qual marcou mais golos, qual sofreu menos e qual tem o melhor aproveitamento de pontos?

# Objetivo da Análise: Construir um ranking completo de desempenho das equipas da La Liga Espanhola, calculando vitórias, empates, derrotas (em casa e fora), golos marcados, golos sofridos, diferença de golos e pontos conquistados, para identificar as equipas mais dominantes e as mais vulneráveis ao longo das 4 épocas.

# Métricas: vitorias, empates, derrotas, golos_marcados, golos_sofridos, diferenca_golos, pontos, pct_aproveitamento

# Pontos e resultados por equipa em casa
df_es_home = (df_es.groupBy("HomeTeam")
    .agg(
        F.sum(F.when(F.col("FTR") == "H", 1).otherwise(0)).alias("vitorias_casa"),
        F.sum(F.when(F.col("FTR") == "D", 1).otherwise(0)).alias("empates_casa"),
        F.sum(F.when(F.col("FTR") == "A", 1).otherwise(0)).alias("derrotas_casa"),
        F.sum("FTHG").alias("golos_marcados_casa"),
        F.sum("FTAG").alias("golos_sofridos_casa"),
        F.sum(F.when(F.col("FTR") == "H", 3).when(F.col("FTR") == "D", 1).otherwise(0)).alias("pontos_casa"),
        F.count("*").alias("jogos_casa")
    )
)

# Pontos e resultados por equipa fora
df_es_away = (df_es.groupBy("AwayTeam")
    .agg(
        F.sum(F.when(F.col("FTR") == "A", 1).otherwise(0)).alias("vitorias_fora"),
        F.sum(F.when(F.col("FTR") == "D", 1).otherwise(0)).alias("empates_fora"),
        F.sum(F.when(F.col("FTR") == "H", 1).otherwise(0)).alias("derrotas_fora"),
        F.sum("FTAG").alias("golos_marcados_fora"),
        F.sum("FTHG").alias("golos_sofridos_fora"),
        F.sum(F.when(F.col("FTR") == "A", 3).when(F.col("FTR") == "D", 1).otherwise(0)).alias("pontos_fora"),
        F.count("*").alias("jogos_fora")
    )
)

# Join e cálculo de métricas totais
df_es_ranking = (df_es_home.join(df_es_away,
        df_es_home.HomeTeam == df_es_away.AwayTeam, "inner")
    .select(
        F.col("HomeTeam").alias("equipa"),
        (F.col("vitorias_casa") + F.col("vitorias_fora")).alias("vitorias"),
        (F.col("empates_casa") + F.col("empates_fora")).alias("empates"),
        (F.col("derrotas_casa") + F.col("derrotas_fora")).alias("derrotas"),
        (F.col("golos_marcados_casa") + F.col("golos_marcados_fora")).alias("golos_marcados"),
        (F.col("golos_sofridos_casa") + F.col("golos_sofridos_fora")).alias("golos_sofridos"),
        (F.col("pontos_casa") + F.col("pontos_fora")).alias("pontos"),
        (F.col("jogos_casa") + F.col("jogos_fora")).alias("jogos_totais")
    )
    .withColumn("diferenca_golos", F.col("golos_marcados") - F.col("golos_sofridos"))
    .withColumn("pct_aproveitamento", F.round(F.col("pontos") / (F.col("jogos_totais") * 3) * 100, 1))
    .orderBy(F.desc("pontos"))
)

print("=== Ranking de Desempenho — La Liga Espanhola (4 Épocas) ===")
print("Ordenado por pontos conquistados (Vitória=3pts, Empate=1pt, Derrota=0pts)\n")
display(df_es_ranking)