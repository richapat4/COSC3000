% COSC3000 - Computer Graphics and Data Visualisation
%
% Prac week 1
% Univariate Data Visualisation


%% PLOTS / SUBPLOTS 

clear all; 
close all;

x = -pi : 0.01: pi;
%pre-allocate space for matrix; 
data = zeros([length(x) 3]); % row columns

data(:,1) = sin(x);
data(:,2) = cos(x);
data(:,3) = log(abs(x));

figure('Color',[1 1 1]);
plot( data(:,1), '-b'); 
axis([0 length(x) -3 3]);

xlabel('x');
ylabel('y');

title('Data');
hold on; 

plot( data(:,2), '-r'); 
axis([0 length(x) -3 3]);

plot( data(:,3), '-g'); 
axis([0 length(x) -3 3]);

plot(zeros(1, length(x)), ':k'); 


hold off;

%% SUBPLOTS

figure('Color',[1 1 1]);

for i = 1:3
    %code to repeat
    subplot(3,1,i);
    plot( data(:,i), '-b'); 
    hold on;
    axis([0 length(x) -3 3]);
    plot(zeros(1, length(x)), ':k'); 
    xlabel('x');
    ylabel('y');
    title("Data");
end

hold off;
%% CELL ARRAYS 


temperature(1  ,:) = {'01-Jan-2010', [45,49,0]};

temperature(2  ,:) = {'03-Apr-2010', [54,68,21]};

temperature(3  ,:) = {'20-Jun-2010', [72,85,53]};

temperature(4  ,:) = {'15-Sep-2010', [63,81,56]};

temperature(5  ,:) = {'31-Dec-2010', [38,54,18]};

allTemps = cell2mat(temperature(:,2));
dates = datenum(temperature(:,1), 'dd-mmm-yyyy');

plot(dates, allTemps);
datetick('x', 'mmm');
ylabel('deg F');
legend(['city 1' ; 'city 2' ; 'city 3'])


%% HOSPITAL DATA SET 

load hospital.mat

h1 = hospital(1:10, :);
h2 = hospital(:,{'LastName' 'Age' 'Sex' 'Smoker'});

hospital.AtRisk = hospital.Smoker | (hospital.Age > 40);

%Use individual variables to explore data 
boxplot(hospital.Age, hospital.Sex)
h3 = hospital(hospital.Age < 30, {'LastName' 'Age' 'Sex' 'Smoker'} );

%Sort the observations based on two vars 
h4 = sortrows(hospital, {'LastName' 'Age' 'Sex' 'Smoker'});

%Draw bar chart for number of variables 

vars= {'Age', 'Weight'}

for i = 1: length(vars)
    variable = char(vars(i))
    data = hospital.( variable ) 
    figure;
    bar(data);
    title(vars(i));
end 


% 

%% SETUP; 
close all
clear all

% load the flu sample data set
data = load( 'flu.mat' );

%% HISTOGRAM
% draw a histogram
figure( 'Color', [1 1 1] );
data = load( 'flu.mat' );
histogram(data.flu.NE);


%% QUANTILE PLOT
% draw a quantile plot
figure( 'Color', [1 1 1] );
data = load( 'flu.mat' );
qqplot(data.flu.MidAtl);


%% Q-Q PLOT
% draw a q-q plot to compare, for example, MidAtl with SAtl
figure( 'Color', [1 1 1] );
ret = qqplot(data.flu.MidAtl);
hold on; 
ret = qqplot(data.flu.SAtl);




%% TUKEY MEAN DIFFERENCE PLOT
% draw a mean difference plot to compare, for example, NE with WSCentral
figure( 'Color', [1 1 1] );



%% HEATMAP
% draw a heatmap

cdata = data.flu.MidAtl;
h = heatmap(cdata);


%% BAR CHARTS
% draw a bar chart
figure( 'Color', [1 1 1] )

subplot(2,1,1);
bar(data.flu.NE);

subplot(2,1,2);
barh(data.flu.MidAtl);


%% STAIRSTEP PLOT
% draw a stairstep plot using the stairs function
figure( 'Color', [1 1 1] );

stairs(data.flu.MidAtl);



%% STEM PLOT
% draw a stem plot using the stem function
figure( 'Color', [1 1 1] );
stem(data.flu.MidAtl);



%% SPARKLINE
% draw a series of sparklines using the subplot and plot functions
figure( 'Color', [1 1 1] );



%% TRENDLINE
% draw a trendline, in this case a moving average of 3 points, on top of
% sparklines, using the tsmovavg function
figure( 'Color', [1 1 1] );



%% STREAMGRAPH
% draw a series of streamgraphs, ensuring to half the values and plot both
% the values and their inverse
figure( 'Color', [1 1 1] );



%% JITTER PLOT
% draw a series of jitter plots using scatter instead of plot
figure( 'Color', [1 1 1] );



%% BOX PLOT
% draw a series of box plots using boxplot instead of plot
figure( 'Color', [1 1 1] );
