import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns

# Step 1: Data Loading & Initial Exploration 
df = pd.read_excel('/Users/hamza/Desktop/Work/Hotel_reservations_summary_statistics.xlsx')
print ('Data Shape : ')
print (df.shape)
print(' ================================================ ')
print('Data info :')
print(df.info())


# Step 2: Great useful features 
df['total_nights'] = df['no_of_weekend_nights'] + df ['no_of_week_nights']
df['total_guests'] = df ['no_of_adults'] + df['no_of_children']
df['booking_value'] = df['avg_price_per_room'] + df['total_nights']


# Convert to datetime (handle the float values)
df['arrival_date'] = df['arrival_date'].astype(int)
df['arrival_month'] = df['arrival_month'].astype(int)
df['arrival_year'] = df['arrival_year'].astype(int)

# Create full date string with error handling
def create_date(row):
    try:
        # First try to create the date
        date_str = f"{int(row['arrival_year'])}-{int(row['arrival_month'])}-{int(row['arrival_date'])}"
        return pd.to_datetime(date_str, errors='coerce')
    except:
        return pd.NaT

df['full_date'] = df.apply(create_date, axis=1)

# Check for invalid dates
invalid_dates = df[df['full_date'].isna()]
print(f"\nNumber of invalid dates: {len(invalid_dates)}")
if len(invalid_dates) > 0:
    print("\nInvalid date samples:")
    print(invalid_dates[['arrival_year', 'arrival_month', 'arrival_date']].head())

# Extract day of the week from valid dates only
df['day_of_week'] = df['full_date'].dt.dayofweek

# Map to day names
day_names = {
    0: 'Monday',
    1: 'Tuesday',
    2: 'Wednesday',
    3: 'Thursday',
    4: 'Friday',
    5: 'Saturday',
    6: 'Sunday'
}
df['day_name'] = df['day_of_week'].map(day_names)

# For invalid dates, we'll fill with mode or drop
if len(invalid_dates) > 0:
    # Option 1: Fill with mode (most common day)
    mode_day = df['day_name'].mode()[0]
    df['day_name'] = df['day_name'].fillna(mode_day)
    
    # Option 2: Drop invalid rows (if few)
    # df = df.dropna(subset=['full_date'])
    
    print(f"\nFilled {len(invalid_dates)} invalid dates with mode: {mode_day}")

# Now analyze!
day_of_week_bookings = df['day_name'].value_counts()
print("\n=== DAY OF WEEK BOOKINGS ===")
print(day_of_week_bookings)

#Step 3: Revenue & Business Performance Analysis
# 1. Cancellation impact analysis
cancellation_rate = df['booking_status'].value_counts(normalize=True)
total_lost_revenue = df[df['booking_status']== 'Canceled' ]['booking_value'].sum()
# 2. Monthly revenue trends
monthly_revenue = df.groupby('arrival_month')['booking_value'].sum()
monthly_bookings = df.groupby('arrival_month').size()
# 3. Room type profitability
room_type_profit = df.groupby('room_type_reserved')['booking_value'].mean()

#Step 4: Guest Behavior Analysis
# 1. Lead time distribution
lead_time_stats = df['lead_time'].describe()
# 2. Stay duration patterns
Stay_duration_patterns = df['total_nights'].describe()
# 3. Guest composition analysis
adult_child_ratio = df['no_of_adults'].mean() / (df['no_of_children'].mean() + 0.001)

# Step 5: Cancellation Risk Factors
# Compare canceled vs not canceled bookings
canceled_df = df[df['booking_status'] == 'Canceled' ]
not_canceled_df = df[df['booking_status'] == 'Not_Canceled' ]
# differences in features
features_comprison = pd.DataFrame ({
        'Canceled_Mean' : canceled_df.mean(numeric_only=True),
        'Not_Canceled_Mean' : not_canceled_df.mean(numeric_only=True),
        'Difference': canceled_df.mean(numeric_only=True) - not_canceled_df.mean(numeric_only=True)
})

# Step 6: Market Segment Analysis
# Value by market segment
segment_value = df.groupby('market_segment_type').agg({
    'booking_value': ['mean', 'sum', 'count'],
    'lead_time': 'mean',
    'repeated_guest': 'mean'
})

#Step 7: Visualization & Reporting
# DAY OF WEEK VISUALIZATION
print("\n" + "="*60)
print("DAY OF WEEK ANALYSIS")
print("="*60)

# 1. Bookings by Day of Week
plt.figure(figsize=(12, 8))

# Create subplots
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# 1.1 Total Bookings by Day (Bar Chart)
days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
day_counts = df['day_name'].value_counts().reindex(days_order)

ax1 = axes[0, 0]
bars = ax1.bar(day_counts.index, day_counts.values, color='skyblue', edgecolor='black')
ax1.set_title('Total Bookings by Day of Week', fontsize=14, fontweight='bold')
ax1.set_xlabel('Day of Week', fontsize=12)
ax1.set_ylabel('Number of Bookings', fontsize=12)
ax1.tick_params(axis='x', rotation=45)
ax1.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
             f'{int(height)}', ha='center', va='bottom', fontsize=10)

# 1.2 Bookings by Day (Pie Chart)
ax2 = axes[0, 1]
colors = plt.cm.Set3(np.linspace(0, 1, 7))
wedges, texts, autotexts = ax2.pie(day_counts.values, labels=day_counts.index, 
                                   autopct='%1.1f%%', colors=colors, startangle=90)
ax2.set_title('Booking Distribution by Day', fontsize=14, fontweight='bold')
ax2.axis('equal')  # Equal aspect ratio ensures pie is drawn as a circle

# 1.3 Cancellation Rate by Day
ax3 = axes[0, 2]
cancellation_by_day = df.groupby('day_name')['booking_status'].apply(
    lambda x: (x == 'Canceled').mean() * 100
).reindex(days_order)

bars3 = ax3.bar(cancellation_by_day.index, cancellation_by_day.values, 
                color=['#ff6b6b' if val > cancellation_by_day.mean() else '#4ecdc4' for val in cancellation_by_day.values],
                edgecolor='black')
ax3.set_title('Cancellation Rate by Day (%)', fontsize=14, fontweight='bold')
ax3.set_xlabel('Day of Week', fontsize=12)
ax3.set_ylabel('Cancellation Rate %', fontsize=12)
ax3.tick_params(axis='x', rotation=45)
ax3.axhline(y=cancellation_by_day.mean(), color='red', linestyle='--', alpha=0.7, 
            label=f'Avg: {cancellation_by_day.mean():.1f}%')
ax3.legend()
ax3.grid(axis='y', alpha=0.3)

# Add value labels
for bar in bars3:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
             f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

# 1.4 Average Revenue by Day
ax4 = axes[1, 0]
revenue_by_day = df.groupby('day_name')['booking_value'].mean().reindex(days_order)

bars4 = ax4.bar(revenue_by_day.index, revenue_by_day.values, color='lightgreen', edgecolor='black')
ax4.set_title('Average Revenue per Booking by Day', fontsize=14, fontweight='bold')
ax4.set_xlabel('Day of Week', fontsize=12)
ax4.set_ylabel('Average Revenue ($)', fontsize=12)
ax4.tick_params(axis='x', rotation=45)
ax4.axhline(y=revenue_by_day.mean(), color='red', linestyle='--', alpha=0.7, 
            label=f'Avg: ${revenue_by_day.mean():.0f}')
ax4.legend()
ax4.grid(axis='y', alpha=0.3)

# Add value labels
for bar in bars4:
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + 5,
             f'${height:.0f}', ha='center', va='bottom', fontsize=9)

# 1.5 Total Nights by Day
ax5 = axes[1, 1]
nights_by_day = df.groupby('day_name')['total_nights'].mean().reindex(days_order)

bars5 = ax5.bar(nights_by_day.index, nights_by_day.values, color='orange', edgecolor='black')
ax5.set_title('Average Stay Duration by Day', fontsize=14, fontweight='bold')
ax5.set_xlabel('Day of Week', fontsize=12)
ax5.set_ylabel('Average Nights', fontsize=12)
ax5.tick_params(axis='x', rotation=45)
ax5.axhline(y=nights_by_day.mean(), color='red', linestyle='--', alpha=0.7, 
            label=f'Avg: {nights_by_day.mean():.1f} nights')
ax5.legend()
ax5.grid(axis='y', alpha=0.3)

# Add value labels
for bar in bars5:
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2., height + 0.05,
             f'{height:.1f}', ha='center', va='bottom', fontsize=9)

# 1.6 Lead Time by Day
ax6 = axes[1, 2]
lead_time_by_day = df.groupby('day_name')['lead_time'].mean().reindex(days_order)

bars6 = ax6.bar(lead_time_by_day.index, lead_time_by_day.values, color='purple', alpha=0.7, edgecolor='black')
ax6.set_title('Average Lead Time by Day', fontsize=14, fontweight='bold')
ax6.set_xlabel('Day of Week', fontsize=12)
ax6.set_ylabel('Days in Advance', fontsize=12)
ax6.tick_params(axis='x', rotation=45)
ax6.axhline(y=lead_time_by_day.mean(), color='red', linestyle='--', alpha=0.7, 
            label=f'Avg: {lead_time_by_day.mean():.0f} days')
ax6.legend()
ax6.grid(axis='y', alpha=0.3)

# Add value labels
for bar in bars6:
    height = bar.get_height()
    ax6.text(bar.get_x() + bar.get_width()/2., height + 2,
             f'{height:.0f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()

# 2. Heatmap: Day of Week vs Arrival Month
print("\n" + "="*60)
print("DAY OF WEEK HEATMAP ANALYSIS")
print("="*60)

# Create pivot table for heatmap
heatmap_data = pd.crosstab(df['day_name'], df['arrival_month'])
heatmap_data = heatmap_data.reindex(days_order)  # Reorder days

plt.figure(figsize=(12, 6))
sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='YlOrRd', linewidths=1, linecolor='gray')
plt.title('Bookings Heatmap: Day of Week vs Month', fontsize=16, fontweight='bold')
plt.xlabel('Month', fontsize=12)
plt.ylabel('Day of Week', fontsize=12)
plt.tight_layout()
plt.show()

# 3. Combined Day Analysis - Single Comprehensive Plot
plt.figure(figsize=(15, 10))

# Create normalized metrics for comparison
metrics = pd.DataFrame({
    'Bookings': day_counts / day_counts.max(),
    'Cancellation_Rate': cancellation_by_day / cancellation_by_day.max(),
    'Revenue': revenue_by_day / revenue_by_day.max(),
    'Stay_Length': nights_by_day / nights_by_day.max(),
    'Lead_Time': lead_time_by_day / lead_time_by_day.max()
})

# Line plot for comparison
plt.plot(metrics.index, metrics['Bookings'], marker='o', linewidth=3, markersize=8, label='Bookings (Norm)')
plt.plot(metrics.index, metrics['Revenue'], marker='s', linewidth=3, markersize=8, label='Revenue (Norm)')
plt.plot(metrics.index, metrics['Cancellation_Rate'], marker='^', linewidth=3, markersize=8, label='Cancellation Rate (Norm)')
plt.plot(metrics.index, metrics['Stay_Length'], marker='d', linewidth=3, markersize=8, label='Stay Length (Norm)')
plt.plot(metrics.index, metrics['Lead_Time'], marker='*', linewidth=3, markersize=8, label='Lead Time (Norm)')

plt.title('Day-of-Week Performance Metrics (Normalized)', fontsize=16, fontweight='bold')
plt.xlabel('Day of Week', fontsize=14)
plt.ylabel('Normalized Value', fontsize=14)
plt.legend(loc='best', fontsize=12)
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
