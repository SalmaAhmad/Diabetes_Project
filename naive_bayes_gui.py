import tkinter as tk
from tkinter import ttk, messagebox, font
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score


class NaiveBayesGUI:
    def __init__(self, root, nb_model, X_train, y_train, X_test, y_test, scaler, num_col):
        self.root = root
        self.root.title("🔬 Diabetes Prediction System - Naive Bayes")
        self.root.geometry("1300x800")
        self.root.configure(bg='#2c3e50')

        # Store model and data
        self.nb_model = nb_model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.scaler = scaler
        self.num_col = num_col

        # Calculate metrics
        self.calculate_metrics()

        # Set color scheme
        self.colors = {
            'primary': '#3498db',
            'secondary': '#2c3e50',
            'success': '#27ae60',
            'warning': '#f39c12',
            'danger': '#e74c3c',
            'light': '#ecf0f1',
            'dark': '#34495e',
            'info': '#1abc9c',
            'purple': '#9b59b6',
            'background': '#2c3e50',
            'card_bg': '#34495e',
            'text_light': '#ecf0f1',
            'text_dark': '#2c3e50'
        }

        # Custom fonts
        self.fonts = {
            'title': ('Segoe UI', 22, 'bold'),
            'heading': ('Segoe UI', 14, 'bold'),
            'subheading': ('Segoe UI', 11, 'bold'),
            'normal': ('Segoe UI', 10),
            'small': ('Segoe UI', 9)
        }

        # Create GUI
        self.setup_ui()

    def calculate_metrics(self):
        """Calculate model metrics"""
        y_pred = self.nb_model.predict(self.X_test)
        self.accuracy = accuracy_score(self.y_test, y_pred)
        self.precision = precision_score(self.y_test, y_pred, average='weighted')
        self.recall = recall_score(self.y_test, y_pred, average='weighted')
        self.cm = confusion_matrix(self.y_test, y_pred, labels=['N', 'P', 'Y'])
        self.class_acc = np.diag(self.cm) / self.cm.sum(axis=1)

    def setup_ui(self):
        """Setup the GUI interface with beautiful theme"""
        # Main container
        main_frame = tk.Frame(self.root, bg=self.colors['secondary'], padx=20, pady=20)
        main_frame.pack(fill='both', expand=True)

        # Title Section with icon
        title_frame = tk.Frame(main_frame, bg=self.colors['secondary'])
        title_frame.pack(fill='x', pady=(0, 20))

        title_label = tk.Label(
            title_frame,
            text="🧬 Diabetes Prediction System",
            font=self.fonts['title'],
            fg=self.colors['light'],
            bg=self.colors['secondary']
        )
        title_label.pack()

        subtitle_label = tk.Label(
            title_frame,
            text="Powered by Gaussian Naive Bayes Machine Learning Model",
            font=self.fonts['small'],
            fg=self.colors['info'],
            bg=self.colors['secondary']
        )
        subtitle_label.pack(pady=(5, 0))

        # Main content container
        content_frame = tk.Frame(main_frame, bg=self.colors['secondary'])
        content_frame.pack(fill='both', expand=True)

        # Left Frame - Model Metrics (Colorful Cards)
        left_frame = tk.LabelFrame(
            content_frame,
            text="📊 Model Performance",
            font=self.fonts['heading'],
            bg=self.colors['card_bg'],
            fg=self.colors['light'],
            padx=20,
            pady=20,
            relief='solid',
            borderwidth=2
        )
        left_frame.grid(row=0, column=0, padx=(0, 10), pady=10, sticky='nsew')

        self.create_metrics_cards(left_frame)

        # Center Frame - Patient Input (Modern Form)
        center_frame = tk.LabelFrame(
            content_frame,
            text="👤 Patient Information",
            font=self.fonts['heading'],
            bg=self.colors['card_bg'],
            fg=self.colors['light'],
            padx=20,
            pady=20,
            relief='solid',
            borderwidth=2
        )
        center_frame.grid(row=0, column=1, padx=10, pady=10, sticky='nsew')

        self.create_input_form(center_frame)

        # Right Frame - Prediction Results
        self.right_frame = tk.LabelFrame(
            content_frame,
            text="🎯 Prediction Results",
            font=self.fonts['heading'],
            bg=self.colors['card_bg'],
            fg=self.colors['light'],
            padx=20,
            pady=20,
            relief='solid',
            borderwidth=2
        )
        self.right_frame.grid(row=0, column=2, padx=(10, 0), pady=10, sticky='nsew')

        self.initialize_results_display()

        # Bottom Frame - Visualization Tools
        bottom_frame = tk.LabelFrame(
            content_frame,
            text="📈 Visualizations & Tools",
            font=self.fonts['heading'],
            bg=self.colors['card_bg'],
            fg=self.colors['light'],
            padx=20,
            pady=20,
            relief='solid',
            borderwidth=2
        )
        bottom_frame.grid(row=1, column=0, columnspan=3, pady=(20, 0), sticky='ew')

        self.create_visualization_buttons(bottom_frame)

        # Status Bar
        status_frame = tk.Frame(main_frame, bg=self.colors['dark'], height=30)
        status_frame.pack(fill='x', pady=(20, 0))

        status_label = tk.Label(
            status_frame,
            text=f"✅ Ready for predictions | Model Accuracy: {self.accuracy:.1%}",
            font=self.fonts['small'],
            fg=self.colors['success'],
            bg=self.colors['dark']
        )
        status_label.pack(side='left', padx=10)

        # Configure grid weights
        content_frame.grid_columnconfigure(0, weight=1)
        content_frame.grid_columnconfigure(1, weight=1)
        content_frame.grid_columnconfigure(2, weight=1)
        content_frame.grid_rowconfigure(0, weight=1)

        # Set minimum sizes
        content_frame.grid_columnconfigure(0, minsize=350)
        content_frame.grid_columnconfigure(1, minsize=350)
        content_frame.grid_columnconfigure(2, minsize=350)

    def create_metrics_cards(self, parent):
        """Create beautiful metric cards"""
        metrics = [
            ("🎯 Accuracy", self.accuracy, self.colors['primary'], "{:.1%}"),
            ("🎯 Precision", self.precision, self.colors['info'], "{:.1%}"),
            ("🎯 Recall", self.recall, self.colors['purple'], "{:.1%}")
        ]

        for i, (name, value, color, fmt) in enumerate(metrics):
            metric_frame = tk.Frame(parent, bg=self.colors['card_bg'])
            metric_frame.pack(fill='x', pady=10)

            # Metric icon and name
            name_label = tk.Label(
                metric_frame,
                text=name,
                font=self.fonts['subheading'],
                fg=self.colors['light'],
                bg=self.colors['card_bg']
            )
            name_label.pack(anchor='w')

            # Metric value with color
            value_label = tk.Label(
                metric_frame,
                text=fmt.format(value),
                font=('Segoe UI', 20, 'bold'),
                fg=color,
                bg=self.colors['card_bg']
            )
            value_label.pack(anchor='w', pady=(5, 0))

        # Separator
        separator = ttk.Separator(parent, orient='horizontal')
        separator.pack(fill='x', pady=20)

        # Per-class accuracy
        class_frame = tk.Frame(parent, bg=self.colors['card_bg'])
        class_frame.pack(fill='x')

        class_label = tk.Label(
            class_frame,
            text="🎯 Per-Class Accuracy:",
            font=self.fonts['subheading'],
            fg=self.colors['light'],
            bg=self.colors['card_bg']
        )
        class_label.pack(anchor='w', pady=(0, 10))

        classes = ['N (Normal)', 'P (Prediabetes)', 'Y (Diabetes)']
        class_colors = [self.colors['success'], self.colors['warning'], self.colors['danger']]

        for i, (cls_name, acc, color) in enumerate(zip(classes, self.class_acc, class_colors)):
            class_row = tk.Frame(class_frame, bg=self.colors['card_bg'])
            class_row.pack(fill='x', pady=3)

            cls_label = tk.Label(
                class_row,
                text=f"• {cls_name}:",
                font=self.fonts['normal'],
                fg=self.colors['light'],
                bg=self.colors['card_bg']
            )
            cls_label.pack(side='left')

            acc_label = tk.Label(
                class_row,
                text=f"{acc:.1%}",
                font=('Segoe UI', 11, 'bold'),
                fg=color,
                bg=self.colors['card_bg']
            )
            acc_label.pack(side='right')

    def create_input_form(self, parent):
        """Create the patient input form with modern design"""
        # Form container with scrollbar
        form_container = tk.Frame(parent, bg=self.colors['card_bg'])
        form_container.pack(fill='both', expand=True)

        # Create scrollable canvas
        canvas = tk.Canvas(form_container, bg=self.colors['card_bg'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(form_container, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=self.colors['card_bg'])

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.entries = {}
        default_values = {
            'AGE': 45, 'Urea': 5.0, 'Cr': 70, 'HbA1c': 5.5,
            'Chol': 4.5, 'TG': 1.2, 'HDL': 1.3, 'LDL': 2.5,
            'VLDL': 0.8, 'BMI': 25.0
        }

        # Feature descriptions
        descriptions = {
            'AGE': "Patient age in years",
            'Urea': "Blood urea nitrogen level (mmol/L)",
            'Cr': "Creatinine level (μmol/L)",
            'HbA1c': "Glycated hemoglobin (%)",
            'Chol': "Total cholesterol (mmol/L)",
            'TG': "Triglycerides (mmol/L)",
            'HDL': "High-density lipoprotein (mmol/L)",
            'LDL': "Low-density lipoprotein (mmol/L)",
            'VLDL': "Very low-density lipoprotein (mmol/L)",
            'BMI': "Body Mass Index (kg/m²)"
        }

        for i, feature in enumerate(self.num_col):
            # Feature container with styling
            feature_frame = tk.Frame(scrollable_frame, bg=self.colors['card_bg'])
            feature_frame.pack(fill='x', pady=8)

            # Feature label with description
            label_text = f"🔹 {feature}: {descriptions.get(feature, '')}"
            label = tk.Label(
                feature_frame,
                text=label_text,
                font=self.fonts['normal'],
                fg=self.colors['light'],
                bg=self.colors['card_bg'],
                wraplength=250,
                justify='left'
            )
            label.pack(anchor='w', padx=10, pady=(10, 5))

            # Input entry with placeholder
            entry_frame = tk.Frame(feature_frame, bg=self.colors['card_bg'])
            entry_frame.pack(fill='x', padx=10, pady=(0, 10))

            entry = tk.Entry(
                entry_frame,
                width=25,
                font=self.fonts['normal'],
                bg=self.colors['light'],
                fg=self.colors['dark'],
                relief='flat',
                insertbackground=self.colors['dark']
            )
            entry.insert(0, str(default_values[feature]))
            entry.pack(side='left', fill='x', expand=True, padx=(0, 10))

            # Unit label
            unit = tk.Label(
                entry_frame,
                text=self.get_unit(feature),
                font=self.fonts['small'],
                fg=self.colors['info'],
                bg=self.colors['card_bg']
            )
            unit.pack(side='right')

            self.entries[feature] = entry

        # Predict button with modern design
        button_frame = tk.Frame(scrollable_frame, bg=self.colors['card_bg'])
        button_frame.pack(pady=30)

        predict_btn = tk.Button(
            button_frame,
            text="🚀 Predict Diabetes Risk",
            command=self.predict,
            font=self.fonts['subheading'],
            bg=self.colors['primary'],
            fg='white',
            padx=20,
            pady=10,
            relief='flat',
            cursor="hand2",
            activebackground='#2980b9',
            activeforeground='white'
        )
        predict_btn.pack(pady=10)

        # Clear button
        clear_btn = tk.Button(
            button_frame,
            text="🗑️ Clear Form",
            command=self.clear_form,
            font=self.fonts['subheading'],
            bg=self.colors['info'],
            fg='white',
            padx=20,
            pady=8,
            relief='flat',
            cursor="hand2",
            activebackground='#16a085',
            activeforeground='white'
        )
        clear_btn.pack()

    def get_unit(self, feature):
        """Get unit for each feature"""
        units = {
            'AGE': 'years',
            'Urea': 'mmol/L',
            'Cr': 'μmol/L',
            'HbA1c': '%',
            'Chol': 'mmol/L',
            'TG': 'mmol/L',
            'HDL': 'mmol/L',
            'LDL': 'mmol/L',
            'VLDL': 'mmol/L',
            'BMI': 'kg/m²'
        }
        return units.get(feature, '')

    def initialize_results_display(self):
        """Initialize the results display area"""
        # Result icon and text
        self.result_icon = tk.StringVar()
        self.result_icon.set("⏳")

        self.result_text = tk.StringVar()
        self.result_text.set("Awaiting prediction...\n\nEnter patient data and click 'Predict'")

        # Store current color
        self.current_color = self.colors['light']

        # Icon label
        icon_label = tk.Label(
            self.right_frame,
            textvariable=self.result_icon,
            font=('Segoe UI', 60),
            bg=self.colors['card_bg'],
            fg=self.current_color
        )
        icon_label.pack(pady=(10, 20))

        # Result text label
        self.result_label = tk.Label(
            self.right_frame,
            textvariable=self.result_text,
            font=self.fonts['heading'],
            bg=self.colors['card_bg'],
            fg=self.current_color,
            wraplength=300,
            justify='center'
        )
        self.result_label.pack(pady=(0, 20))

        # Probability frame
        self.prob_frame = tk.Frame(self.right_frame, bg=self.colors['card_bg'])
        self.prob_frame.pack(fill='x', pady=10)

        # Confidence bar container
        self.confidence_frame = tk.Frame(self.right_frame, bg=self.colors['card_bg'])
        self.confidence_frame.pack(fill='x', pady=20)

    def create_visualization_buttons(self, parent):
        """Create visualization buttons"""
        viz_buttons = [
            ("📊 Confusion Matrix", self.show_confusion_matrix, self.colors['info']),
            ("📈 Performance Metrics", self.show_metrics_chart, self.colors['primary']),
            ("🎯 Class Accuracy", self.show_class_accuracy, self.colors['success']),

        ]

        button_container = tk.Frame(parent, bg=self.colors['card_bg'])
        button_container.pack()

        for i, (text, command, color) in enumerate(viz_buttons):
            btn = tk.Button(
                button_container,
                text=text,
                command=command,
                font=self.fonts['subheading'],
                bg=color,
                fg='white',
                padx=15,
                pady=8,
                relief='flat',
                cursor="hand2",
                activebackground=self.darken_color(color),
                activeforeground='white'
            )
            btn.grid(row=0, column=i, padx=10)

    def darken_color(self, color):
        """Darken a hex color"""
        # Simple darkening by reducing RGB values
        r = int(color[1:3], 16)
        g = int(color[3:5], 16)
        b = int(color[5:7], 16)

        r = max(0, r - 30)
        g = max(0, g - 30)
        b = max(0, b - 30)

        return f'#{r:02x}{g:02x}{b:02x}'

    def clear_form(self):
        """Clear all input fields"""
        for entry in self.entries.values():
            entry.delete(0, tk.END)

        self.result_icon.set("⏳")
        self.result_text.set("Form cleared!\n\nEnter new patient data and click 'Predict'")
        self.current_color = self.colors['light']
        self.result_label.config(fg=self.current_color)

        # Clear probability display
        for widget in self.prob_frame.winfo_children():
            widget.destroy()

        for widget in self.confidence_frame.winfo_children():
            widget.destroy()

    def predict(self):
        """Make prediction based on user input"""
        try:
            # Collect user input
            user_data = {}
            for feature, entry in self.entries.items():
                value = float(entry.get())
                user_data[feature] = value

            # Create DataFrame with proper column names
            user_df = pd.DataFrame([user_data], columns=self.num_col)

            # Scale the input using the same scaler
            user_scaled = self.scaler.transform(user_df)

            # Convert back to DataFrame with feature names
            user_scaled_df = pd.DataFrame(user_scaled, columns=self.num_col)

            # Make prediction
            prediction = self.nb_model.predict(user_scaled_df)
            probabilities = self.nb_model.predict_proba(user_scaled_df)

            # Update display
            self.update_prediction_display(prediction[0], probabilities[0])

        except ValueError as e:
            messagebox.showerror("Input Error",
                                 f"❌ Please enter valid numeric values for all fields!\n\n"
                                 f"Error: {str(e)}",
                                 icon='error')
    def update_prediction_display(self, prediction, probabilities):
        """Update the results display with prediction"""
        class_info = {
            'N': {'name': 'NORMAL', 'icon': '✅', 'color': self.colors['success'],
                  'description': 'Non-diabetic'},
            'P': {'name': 'PREDIABETES', 'icon': '⚠️', 'color': self.colors['warning'],
                  'description': 'At risk of diabetes'},
            'Y': {'name': 'DIABETES', 'icon': '🚨', 'color': self.colors['danger'],
                  'description': 'Diabetic condition detected'}
        }

        info = class_info[prediction]

        # Update icon and text
        self.result_icon.set(info['icon'])
        self.result_text.set(f"{info['icon']} {info['name']}\n\n{info['description']}")
        self.current_color = info['color']
        self.result_label.config(fg=self.current_color)

        # Update probability bars
        self.update_probability_bars(probabilities)

        # Show recommendation
        self.show_recommendation(prediction)

    def update_probability_bars(self, probabilities):
        """Update probability visualization"""
        # Clear existing widgets
        for widget in self.prob_frame.winfo_children():
            widget.destroy()

        for widget in self.confidence_frame.winfo_children():
            widget.destroy()

        # Create probability bars
        classes = ['N (Normal)', 'P (Prediabetes)', 'Y (Diabetes)']
        colors = [self.colors['success'], self.colors['warning'], self.colors['danger']]

        for i, (cls, prob, color) in enumerate(zip(classes, probabilities, colors)):
            # Probability row
            row_frame = tk.Frame(self.prob_frame, bg=self.colors['card_bg'])
            row_frame.pack(fill='x', pady=5)

            # Class label
            label = tk.Label(
                row_frame,
                text=cls,
                font=self.fonts['small'],
                fg=self.colors['light'],
                bg=self.colors['card_bg'],
                width=15,
                anchor='w'
            )
            label.pack(side='left')

            # Percentage label
            percent_label = tk.Label(
                row_frame,
                text=f"{prob:.1%}",
                font=self.fonts['normal'],
                fg=color,
                bg=self.colors['card_bg'],
                width=8,
                anchor='e'
            )
            percent_label.pack(side='right')

            # Progress bar frame
            bar_frame = tk.Frame(row_frame, bg=self.colors['dark'], height=20)
            bar_frame.pack(side='left', fill='x', expand=True, padx=10)
            bar_frame.pack_propagate(False)

            # Custom progress bar using canvas
            canvas = tk.Canvas(bar_frame, bg=self.colors['dark'], highlightthickness=0, height=20)
            canvas.pack(fill='x')

            # Update canvas after it's drawn
            def update_bar(canvas=canvas, width=bar_frame.winfo_width(), prob=prob, color=color):
                if width > 1:
                    progress_width = int(width * prob)
                    canvas.create_rectangle(0, 0, progress_width, 20, fill=color, outline='')
                    canvas.create_text(progress_width / 2, 10, text=f"{prob:.0%}",
                                       fill='white', font=self.fonts['small'])

            canvas.after(100, update_bar)

    def show_recommendation(self, prediction):
        """Show medical recommendation"""
        recommendations = {
            'N': """✅ EXCELLENT NEWS!

Your results indicate a NORMAL blood sugar profile.

Recommendations:
• Continue healthy lifestyle habits
• Annual checkups recommended
• Maintain balanced diet
• Regular exercise (30 min/day)""",

            'P': """⚠️ ATTENTION REQUIRED!

Your results show PREDIABETIC conditions.

Urgent Actions:
• Consult doctor within 1 week
• Start lifestyle intervention
• Increase physical activity
• Monitor glucose regularly
• Consider dietary changes""",

            'Y': """🚨 IMMEDIATE ACTION NEEDED!

Your results indicate DIABETES.

Critical Steps:
• See doctor within 48 hours
• Start medication as prescribed
• Strict glucose monitoring
• Emergency contact: 911 if severe symptoms
• Follow-up every 3 months"""
        }

        # Create styled messagebox
        msg_window = tk.Toplevel(self.root)
        msg_window.title("Medical Recommendation")
        msg_window.geometry("500x400")
        msg_window.configure(bg=self.colors['card_bg'])
        msg_window.resizable(False, False)

        # Center the window
        msg_window.transient(self.root)
        msg_window.grab_set()

        x = self.root.winfo_x() + (self.root.winfo_width() - 500) // 2
        y = self.root.winfo_y() + (self.root.winfo_height() - 400) // 2
        msg_window.geometry(f"+{x}+{y}")

        # Title
        title = tk.Label(
            msg_window,
            text="💊 Medical Recommendation",
            font=self.fonts['heading'],
            fg=self.colors['light'],
            bg=self.colors['card_bg']
        )
        title.pack(pady=20)

        # Message
        message = tk.Text(
            msg_window,
            wrap=tk.WORD,
            font=self.fonts['normal'],
            bg=self.colors['card_bg'],
            fg=self.colors['light'],
            height=15,
            width=50,
            relief='flat',
            borderwidth=0
        )
        message.insert(1.0, recommendations[prediction])
        message.config(state='disabled')
        message.pack(padx=20, pady=10)

        # Close button
        close_btn = tk.Button(
            msg_window,
            text="Close",
            command=msg_window.destroy,
            font=self.fonts['subheading'],
            bg=self.colors['primary'],
            fg='white',
            padx=20,
            pady=8,
            relief='flat',
            cursor="hand2"
        )
        close_btn.pack(pady=20)

    def show_confusion_matrix(self):
        """Display confusion matrix with theme colors"""
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(7, 6))

        # Custom color map
        cmap = sns.color_palette("Blues", as_cmap=True)

        sns.heatmap(self.cm, annot=True, fmt='d', cmap=cmap,
                    xticklabels=['N', 'P', 'Y'],
                    yticklabels=['N', 'P', 'Y'],
                    ax=ax,
                    annot_kws={"size": 14, "color": "black", "weight": "bold"})

        ax.set_title('Confusion Matrix - Naive Bayes',
                     fontsize=16, fontweight='bold', color='white', pad=20)
        ax.set_xlabel('Predicted Class', fontsize=14, color='white')
        ax.set_ylabel('True Class', fontsize=14, color='white')

        # Set background color
        fig.patch.set_facecolor('#2c3e50')
        ax.set_facecolor('#34495e')

        self.display_plot(fig, "Confusion Matrix")

    def show_metrics_chart(self):
        """Display metrics bar chart with theme"""
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(7, 5))

        metrics = ['Accuracy', 'Precision', 'Recall']
        values = [self.accuracy, self.precision, self.recall]
        colors = [self.colors['success'], self.colors['primary'], self.colors['info']]

        bars = ax.bar(metrics, values, color=colors, edgecolor='white', linewidth=2, alpha=0.8)
        ax.set_ylabel('Score', fontsize=12, color='white')
        ax.set_title('Model Performance Metrics', fontsize=16, fontweight='bold', color='white')
        ax.set_ylim([0, 1.1])
        ax.tick_params(colors='white')

        # Add value labels with glow effect
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                    f'{value:.3f}', ha='center', va='bottom',
                    fontsize=11, fontweight='bold', color='white')

        # Set background
        fig.patch.set_facecolor('#2c3e50')
        ax.set_facecolor('#34495e')

        self.display_plot(fig, "Performance Metrics")

    def show_class_accuracy(self):
        """Display per-class accuracy with theme"""
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(7, 5))

        classes = ['Normal', 'Prediabetes', 'Diabetes']
        colors = [self.colors['success'], self.colors['warning'], self.colors['danger']]

        bars = ax.bar(classes, self.class_acc, color=colors,
                      edgecolor='white', linewidth=2, alpha=0.8)
        ax.set_ylabel('Accuracy', fontsize=12, color='white')
        ax.set_title('Per-Class Accuracy', fontsize=16, fontweight='bold', color='white')
        ax.set_ylim([0, 1.1])
        ax.tick_params(colors='white')

        # Add value labels
        for bar, acc in zip(bars, self.class_acc):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                    f'{acc:.3f}', ha='center', va='bottom',
                    fontsize=11, fontweight='bold', color='white')

        # Set background
        fig.patch.set_facecolor('#2c3e50')
        ax.set_facecolor('#34495e')

        self.display_plot(fig, "Per-Class Accuracy")

    def export_results(self):
        """Export prediction results to file"""
        # Implementation for exporting results
        messagebox.showinfo("Export", "📁 Export feature coming soon!")

    def display_plot(self, fig, title):
        """Display plot in a styled window"""
        plot_window = tk.Toplevel(self.root)
        plot_window.title(f"📈 {title}")
        plot_window.geometry("800x650")
        plot_window.configure(bg=self.colors['card_bg'])

        # Center the window
        x = self.root.winfo_x() + (self.root.winfo_width() - 800) // 2
        y = self.root.winfo_y() + (self.root.winfo_height() - 650) // 2
        plot_window.geometry(f"+{x}+{y}")

        # Canvas for plot
        canvas = FigureCanvasTkAgg(fig, master=plot_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True, padx=20, pady=20)

        # Close button
        close_frame = tk.Frame(plot_window, bg=self.colors['card_bg'])
        close_frame.pack(pady=10)

        close_btn = tk.Button(
            close_frame,
            text="Close",
            command=plot_window.destroy,
            font=self.fonts['subheading'],
            bg=self.colors['primary'],
            fg='white',
            padx=20,
            pady=8,
            relief='flat',
            cursor="hand2"
        )
        close_btn.pack()


# Main function to run the GUI
def run_gui(nb_model, X_train, y_train, X_test, y_test, scaler, num_col):
    """Run the GUI application"""
    root = tk.Tk()
    app = NaiveBayesGUI(root, nb_model, X_train, y_train, X_test, y_test, scaler, num_col)
    root.mainloop()
