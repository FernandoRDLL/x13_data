clear
cls
//This code uses t-distributed innovations to estimate the conditional volatility of permit growth data by state. If Gaussian innovations are preferred, remove the distribution(t) option from both regression models.
//Load file with permit growth data by state. One column should be for date and
//the others for growth by state
use "x13_vals_pct_change_1960_2024.dta", clear

// Make sure the date variable is in a proper format
format date %tm
tsset date, monthly
drop index

//Sample date restriction
keep if date <= ym(2019,12)


// Create a local macro with all variable names
ds
local allvars `r(varlist)'

// Loop through each variable in the list
foreach var of local allvars {
    // Skip the 'date' variable
    if "`var'" == "date"{
        continue
    }
	
	// Apply GARCH model and check for convergence
    capture arch `var' L1.`var', arch(1) garch(1) distribution(t)
    if _rc == 0 {
		predict aux, variance
		gen cvol_gar_`var' = sqrt(aux)
		drop aux
    } 
	else {
        display "GCH model did not converge for `var'"
    }
		
	// Apply GARCH-GJR model
    capture arch `var' L1.`var', arch(1) garch(1) tarch(1) distribution(t)
    if _rc == 0 {
        predict aux, variance
		gen cvol_gjr_`var' = sqrt(aux)
		drop aux
    } 
	else {
        display "GJR model did not converge for `var'"
    }
}

// Save the dataset with conditional volatilities
// Save in dta format
save cvols_x13_vals_pct_change_1960_2024.dta, replace
// Save in csv format
// export delimited using "[output_name].csv", replace

