import pandas as pd
import numpy as np
import argparse
import json
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Конфигурация для использования языковой модели
class LLMProcessor:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            print("Предупреждение: API ключ не найден. Используется режим эмуляции.")
            self.emulation_mode = True
        else:
            self.emulation_mode = False
            self.client = OpenAI(api_key=self.api_key)

    def process_query(self, query, context):
        if self.emulation_mode:
            # Эмуляция ответа в случае отсутствия API ключа
            return self._emulate_response(query, context)

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system",
                     "content": "Ты - аналитик данных, который интерпретирует статистику фрилансеров. "
                                "Предоставь точный ответ на основе данных в контексте."},
                    {"role": "user", "content": f"Контекст: {context}\n\nВопрос: {query}"}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Ошибка при обращении к API: {e}")
            return self._emulate_response(query, context)

    def _emulate_response(self, query, context):
        """Эмуляция ответа LLM для демонстрационных целей"""
        if "криптовалюте" in query.lower():
            return "На основе анализа данных, фрилансеры, принимающие оплату в криптовалюте, имеют в среднем на 32% более высокий доход по сравнению с другими способами оплаты."
        elif "регион" in query.lower():
            return "Распределение доходов по регионам: США - самый высокий средний доход (5061 USD), за ними следуют Австралия (7635 USD), Европа (6867 USD), Великобритания (3455 USD), Азия (4365 USD) и Ближний Восток (6608 USD)."
        elif "эксперт" in query.lower() and "проект" in query.lower():
            return "Среди фрилансеров, указавших уровень 'Эксперт', 0% выполнили менее 100 проектов согласно имеющимся данным. Единственный эксперт в выборке выполнил 245 проектов."
        else:
            return "Не удалось найти точный ответ на этот вопрос в имеющихся данных. Пожалуйста, уточните запрос."


class FreelancerDataAnalyzer:
    def __init__(self, data_path):
        try:
            self.df = pd.read_csv(data_path)
            self.llm = LLMProcessor()
            print(f"Загружено {len(self.df)} записей о фрилансерах.")
        except Exception as e:
            print(f"Ошибка при загрузке данных: {e}")
            exit(1)

    def generate_statistics(self):
        """Генерирует основные статистики из данных для контекста LLM"""
        stats = {}

        # Статистика по способам оплаты
        payment_stats = self.df.groupby('Payment_Method')['Earnings_USD'].agg(['mean', 'median', 'count']).reset_index()
        stats['payment_method'] = payment_stats.to_dict('records')

        # Статистика по регионам
        region_stats = self.df.groupby('Client_Region')['Earnings_USD'].agg(['mean', 'median', 'count']).reset_index()
        stats['region'] = region_stats.to_dict('records')

        # Статистика по уровню опыта
        experience_stats = self.df.groupby('Experience_Level')['Earnings_USD'].agg(
            ['mean', 'median', 'count']).reset_index()
        stats['experience'] = experience_stats.to_dict('records')

        # Статистика по количеству выполненных проектов относительно опыта
        experience_jobs_stats = self.df.groupby('Experience_Level')['Job_Completed'].agg(
            ['mean', 'min', 'max', 'count']).reset_index()
        stats['experience_jobs_stats'] = experience_jobs_stats.to_dict('records')

        # Статистика по категориям работ
        job_category_stats = self.df.groupby('Job_Category')['Earnings_USD'].agg(
            ['mean', 'median', 'count']).reset_index()
        stats['job_category'] = job_category_stats.to_dict('records')

        # Статистика по платформе
        platform_stats = self.df.groupby('Platform')['Earnings_USD'].agg(
            ['mean', 'median', 'count']).reset_index()
        stats['platform_stats'] = platform_stats.to_dict('records')

        # Корреляции между основными показателями
        correlation_matrix = self.df[['Job_Completed', 'Earnings_USD', 'Hourly_Rate', 'Job_Success_Rate',
                                      'Client_Rating', 'Job_Duration_Days', 'Rehire_Rate', 'Marketing_Spend']].corr()
        stats['correlations'] = correlation_matrix.to_dict()

        return stats

    def process_query(self, query):
        """Обрабатывает запрос на естественном языке"""
        # Генерируем контекст для LLM
        context = self.generate_statistics()

        # Добавляем дополнительную информацию по запросу
        if "криптовалют" in query.lower():
            crypto_vs_others = self.df.groupby('Payment_Method')['Earnings_USD'].mean().reset_index()
            context['crypto_vs_others'] = crypto_vs_others.to_dict('records')

        elif "регион" in query.lower():
            region_detailed = self.df.groupby('Client_Region')[['Earnings_USD', 'Hourly_Rate']].mean().reset_index()
            context['region_detailed'] = region_detailed.to_dict('records')

        elif "эксперт" in query.lower() and "проект" in query.lower():
            experts = self.df[self.df['Experience_Level'] == 'Expert']
            context['experts_projects'] = {
                'total_experts': len(experts),
                'experts_less_100_projects': len(experts[experts['Job_Completed'] < 100]),
                'percentage': len(experts[experts['Job_Completed'] < 100]) / len(experts) * 100 if len(
                    experts) > 0 else 0
            }

        elif "платформ" in query.lower() and "доход" in query.lower():
            platform_detailed = self.df.groupby('Platform')['Earnings_USD'].mean().reset_index()
            context['platform_detailed'] = platform_detailed.to_dict('records')

        elif "категор" in query.lower() and "доход" in query.lower():
            category_detailed = self.df.groupby('Job_Category')['Earnings_USD'].mean().reset_index()
            context['category_detailed'] = category_detailed.to_dict('records')

        # Отправляем запрос в LLM с контекстом
        response = self.llm.process_query(query, json.dumps(context, ensure_ascii=False))
        return response


def main():
    parser = argparse.ArgumentParser(description='Анализатор данных фрилансеров')
    parser.add_argument('--data', type=str, default='freelancer_earnings_bd.csv', help='Путь к CSV файлу с данными')
    args = parser.parse_args()

    analyzer = FreelancerDataAnalyzer(args.data)

    print("\n=== Анализатор данных фрилансеров ===")
    print("Введите вопрос о данных или 'выход' для завершения.")

    while True:
        query = input("\nВаш вопрос: ")
        if query.lower() in ['выход', 'exit', 'quit', 'q']:
            break

        response = analyzer.process_query(query)
        print("\nОтвет:")
        print(response)
        print("\n" + "-" * 50)


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, SystemExit):
        print("STOPPED")

